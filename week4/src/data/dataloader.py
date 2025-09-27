
from torch.utils.data import DataLoader
from .dataset import SpamDataset

def getdataloader(csv_path,tokenizer,max_len=128,batch_size=32,shuffle=True):
    dataset = SpamDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)
    return dataloader

