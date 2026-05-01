import os
import sys
import torch
from transformers import AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.reward_model import RewardModel


class PPOTrainingPipeline:
    def __init__(
        self,
        actor_model_name,
        reward_model_path,
        tokenizer,
        output_dir,
        dataset=None,
        lr=1.41e-5,
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=2,
        target_kl=0.1,
        kl_penalty="kl",
        ppo_epochs=1,
        device="cuda",
        log_with=None,
        local_rank=0,
        world_size=1,
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (local_rank == 0)
        os.makedirs(output_dir, exist_ok=True)

        if self.is_main:
            print("Loading Actor model (with Value Head)...")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            actor_model_name,
            trust_remote_code=True,
        )

        if self.is_main:
            print("Loading Reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            actor_model_name,
            trust_remote_code=True,
        )

        if self.is_main:
            print("Loading Reward model...")
        self.reward_model = RewardModel(actor_model_name)
        rm_state_path = os.path.join(reward_model_path, "pytorch_model.bin")
        if os.path.exists(rm_state_path):
            state_dict = torch.load(rm_state_path, map_location="cpu")
            self.reward_model.load_state_dict(state_dict)
            if self.is_main:
                print(f"Loaded reward model weights from {rm_state_path}")
        else:
            if self.is_main:
                print(f"WARNING: Reward model checkpoint not found at {rm_state_path}. Using random weights!")
        self.reward_model.eval().to(device)

        self.ppo_config = PPOConfig(
            model_name=actor_model_name,
            learning_rate=lr,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ppo_epochs=ppo_epochs,
            kl_penalty=kl_penalty,
            target_kl=target_kl,
            log_with=log_with,
        )

        def ppo_collator(data):
            return {key: [d[key] for d in data] for key in data[0]}

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=dataset,
            data_collator=ppo_collator,
        )

        self.gen_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        self.log_history = []

    def train(self, total_ppo_steps=100, max_new_tokens=40):
        dataloader = self.ppo_trainer.dataloader
        step_count = 0

        iterator = tqdm(dataloader, desc="PPO Training", disable=not self.is_main)
        for batch in iterator:
            if step_count >= total_ppo_steps:
                break

            query_tensors = batch["input_ids"]
            query_texts = batch["query"] if isinstance(batch["query"][0], str) else None

            if isinstance(query_tensors[0], str):
                query_tensors = [
                    self.tokenizer.encode(q, return_tensors="pt", add_special_tokens=False).squeeze(0)
                    for q in query_tensors
                ]

            response_tensors = self.ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **self.gen_kwargs,
                max_new_tokens=max_new_tokens,
            )

            response_texts = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            if query_texts is None:
                query_texts = self.tokenizer.batch_decode(query_tensors, skip_special_tokens=True)

            full_texts = []
            for q, r in zip(query_texts, response_texts):
                if r.startswith(q):
                    r = r[len(q):]
                full_texts.append(q + r)

            with torch.no_grad():
                enc = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                rewards = [r.to(torch.float32) for r in self.reward_model(**enc)]

            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            self.ppo_trainer.log_stats(stats, batch, rewards)

            kl = stats.get("objective/kl", stats.get("ppo/policy/kl", 0.0))
            mean_reward = torch.stack(rewards).mean().item()
            self.log_history.append({
                "step": step_count,
                "kl": float(kl),
                "mean_reward": float(mean_reward),
                "loss": float(stats.get("ppo/loss/total", 0.0)),
            })

            if step_count % 10 == 0 and self.is_main:
                print(f"Step {step_count}: KL={float(kl):.4f}, Mean Reward={float(mean_reward):.4f}")

            step_count += 1

        self.save_results()
        self.save_model()

    def save_results(self):
        if not self.is_main:
            return
        log_path = os.path.join(self.output_dir, "ppo_logs.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_history, f, indent=2, ensure_ascii=False)
        print(f"Saved PPO training logs to {log_path}")

        if len(self.log_history) == 0:
            return

        steps = [x["step"] for x in self.log_history]
        rewards = [x["mean_reward"] for x in self.log_history]
        kls = [x["kl"] for x in self.log_history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "tab:blue"
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Mean Reward", color=color1)
        ax1.plot(steps, rewards, color=color1, label="Mean Reward")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("KL Divergence", color=color2)
        ax2.plot(steps, kls, color=color2, label="KL Divergence")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.tight_layout()
        plot_path = os.path.join(self.output_dir, "reward_kl_curve.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved reward & KL curve to {plot_path}")

    def save_model(self):
        if not self.is_main:
            return
        save_path = os.path.join(self.output_dir, "final_ppo_model")
        self.ppo_trainer.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Saved final PPO model to {save_path}")
