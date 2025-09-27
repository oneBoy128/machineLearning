
from torch.utils.data import DataLoader

from .dataset import SpamDataset


def getDataloader(csv_path,tokenizer,max_len = 128, batch_size=32, shuffle=True):
    print(f"开始加载数据，CSV路径：{csv_path}")  # 加这行测试
    dataset = SpamDataset(csv_path,tokenizer,max_len)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader
    