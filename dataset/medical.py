from dataset import baseDataset
import os.path as osp
import torch

class medicalDataset(baseDataset):
    def __init__(self, split:str,path:str):
        assert split in ["train", "test"]
        assert osp.exists(path)
        assert osp.exists(osp.join(path,"label.json"))
        self.split = split
        self.file_path = path
        self.is_tokenized = False
        self._build()

    def _build(self):
        filename = 'train.dat' if self.split == "train" else 'test.dat'
        with open(osp.join(self.file_path, filename), 'r', encoding='utf-8') as file:
            data_raw = [f.strip() for f in file]

        self.labels = []
        self.texts = []

        for line in data_raw:
            if not line:
                continue
            try:
                if self.split == "train":
                    parts = line.split("\t")
                    if len(parts) == 2:
                        label, text = parts
                        self.labels.append(int(label) - 1)
                        self.texts.append(text)
                else:  # test: no label column
                    self.texts.append(line)
                    self.labels.append(0)  # placeholder
            except Exception:
                continue

    @property
    def _len(self):
        return self.__len__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_tokenized:
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }
        else:
            return {"text": self.texts[idx], "labels": self.labels[idx]}


    def tokenize(self, tokenizer):
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=256,
        )

        self.is_tokenized = True

        return self


def collate_fn(batch,pad_token_id=0):

    input_ids_list = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(len(ids) for ids in input_ids_list)

    batch_input_ids = []
    batch_attention_mask = []

    for ids in input_ids_list:
        pad_len = max_len - len(ids)

        padded_ids = ids + [pad_token_id] * pad_len

        mask = [1] * len(ids) + [0] * pad_len

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(mask)


    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
