from dataset import baseDataset
import os.path as osp

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
        self.texts = [] # 修复1：改名叫 texts，防止和循环变量重名

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
        # 如果已经被 Tokenize 了，就返回处理好的 Tensor
        if self.is_tokenized:
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }
        # 否则返回原始文本
        else:
            return {"text": self.texts[idx], "labels": self.labels[idx]}


    def tokenize(self, tokenizer):
        # 因为我们之前在 collate_fn 里写了动态 Padding
        # 这里可以直接 truncate，不需要 padding="max_length"，更省内存！
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=256,
            # 删除了 padding="max_length" 和 .to("cuda")
        )

        self.is_tokenized = True

        # 修复2：一定要 return self，保证它依然是个 Dataset 对象！
        return self
