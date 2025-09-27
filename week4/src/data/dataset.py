import torch
import pandas as pd
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self,csv_path,tokenizer,max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(csv_path)
        self.label = self.data['label_num'].tolist()
        self.text = self.data['processed_message'].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(label,dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }
