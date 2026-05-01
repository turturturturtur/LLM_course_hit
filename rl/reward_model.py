import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoConfig, AutoModel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import json


class RewardModel(nn.Module):
    def __init__(self, base_model_name, trust_remote_code=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        self.transformer = AutoModel.from_pretrained(
            base_model_name,
            config=self.config,
            trust_remote_code=trust_remote_code,
        )
        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.score = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.score = self.score.to(self.transformer.dtype)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            pooled = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]
        else:
            pooled = hidden_states[:, -1]

        pooled = self.dropout(pooled)
        reward = self.score(pooled).squeeze(-1)
        return reward


def compute_ranking_loss(reward_chosen, reward_rejected):
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss


class RewardModelTrainer:
    def __init__(self, model, tokenizer, output_dir, lr=1e-5, batch_size=4, device="cuda",
                 local_rank=0, world_size=1):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (local_rank == 0)
        self.batch_size = batch_size
        self.global_step = 0
        self.history = []

        self.model = model.to(device)
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        os.makedirs(output_dir, exist_ok=True)

    def _unwrap_model(self):
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def train(self, train_dataset, eval_dataset=None, epochs=1, max_steps=-1):
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                shuffle=False,
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            train_sampler = None

        self.model.train()
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(
                train_loader,
                desc=f"RM Epoch {epoch+1}/{epochs}",
                disable=not self.is_main,
            )
            for step, batch in enumerate(pbar):
                if 0 < max_steps <= step:
                    break

                chosen_ids = batch["input_ids_chosen"].to(self.device)
                chosen_mask = batch["attention_mask_chosen"].to(self.device)
                rejected_ids = batch["input_ids_rejected"].to(self.device)
                rejected_mask = batch["attention_mask_rejected"].to(self.device)

                reward_chosen = self.model(chosen_ids, chosen_mask)
                reward_rejected = self.model(rejected_ids, rejected_mask)

                loss = compute_ranking_loss(reward_chosen, reward_rejected)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.global_step += 1

                with torch.no_grad():
                    correct += (reward_chosen > reward_rejected).sum().item()
                    total += len(reward_chosen)

                acc = correct / total if total > 0 else 0
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
                self.history.append({"step": self.global_step, "loss": loss.item(), "acc": acc})

            if self.world_size > 1:
                epoch_loss_t = torch.tensor(epoch_loss, device=self.device)
                correct_t = torch.tensor(correct, device=self.device)
                total_t = torch.tensor(total, device=self.device)
                num_batches_t = torch.tensor(len(train_loader), device=self.device)
                dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_batches_t, op=dist.ReduceOp.SUM)
                avg_loss = epoch_loss_t.item() / num_batches_t.item() if num_batches_t.item() > 0 else 0
                avg_acc = correct_t.item() / total_t.item() if total_t.item() > 0 else 0
            else:
                avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_acc = correct / total if total > 0 else 0

            if self.is_main:
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

            self.save_checkpoint(epoch + 1)

            if eval_dataset is not None:
                self.evaluate(eval_dataset)

        if self.is_main:
            hist_path = os.path.join(self.output_dir, "rm_history.json")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            print(f"Saved RM training history to {hist_path}")

    def evaluate(self, eval_dataset):
        if self.world_size > 1:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                sampler=eval_sampler,
                shuffle=False,
            )
        else:
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        eval_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating RM", disable=not self.is_main):
                chosen_ids = batch["input_ids_chosen"].to(self.device)
                chosen_mask = batch["attention_mask_chosen"].to(self.device)
                rejected_ids = batch["input_ids_rejected"].to(self.device)
                rejected_mask = batch["attention_mask_rejected"].to(self.device)

                reward_chosen = self.model(chosen_ids, chosen_mask)
                reward_rejected = self.model(rejected_ids, rejected_mask)

                loss = compute_ranking_loss(reward_chosen, reward_rejected)
                eval_loss += loss.item()
                num_batches += 1

                correct += (reward_chosen > reward_rejected).sum().item()
                total += len(reward_chosen)

        if self.world_size > 1:
            eval_loss_t = torch.tensor(eval_loss, device=self.device)
            correct_t = torch.tensor(correct, device=self.device)
            total_t = torch.tensor(total, device=self.device)
            num_batches_t = torch.tensor(num_batches, device=self.device)
            dist.all_reduce(eval_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_t, op=dist.ReduceOp.SUM)
            eval_loss = eval_loss_t.item()
            correct = correct_t.item()
            total = total_t.item()
            num_batches = num_batches_t.item()

        avg_loss = eval_loss / num_batches if num_batches > 0 else 0
        avg_acc = correct / total if total > 0 else 0
        if self.is_main:
            print(f"[Eval] Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        return avg_acc

    def save_checkpoint(self, epoch):
        if not self.is_main:
            return
        path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(path, exist_ok=True)
        unwrapped = self._unwrap_model()
        torch.save(unwrapped.state_dict(), os.path.join(path, "pytorch_model.bin"))
        unwrapped.config.save_pretrained(path)
        print(f"Saved checkpoint to {path}")

    def save_model(self, path=None):
        if not self.is_main:
            return
        if path is None:
            path = os.path.join(self.output_dir, "final_model")
        os.makedirs(path, exist_ok=True)
        unwrapped = self._unwrap_model()
        torch.save(unwrapped.state_dict(), os.path.join(path, "pytorch_model.bin"))
        unwrapped.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved final reward model to {path}")
