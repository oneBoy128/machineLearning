import torch
import pandas as pd
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = self.data['processed_message'].tolist()
        self.label = self.data['label_num'].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]

        encodeing = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encodeing['input_ids'].squeeze(0)
        attention_mask = encodeing['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label,dtype=torch.long),
        }
