
from transformers import BertTokenizer

from ..data.dataloader import getDataloader

CSV_PATH = '/home/wby/projects/week4(2)/data/final.csv'
BERT_PATH = '/home/wby/projects/model/bert-base-cased'

def testfunc():
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    dataloader = getDataloader(
        csv_path = CSV_PATH,
        tokenizer = tokenizer,
        batch_size = 8,
        shuffle = False,
    )

    for batch in dataloader:
        print(f"形状为: {batch['input_ids'].shape}")
        print(f'编码形状: {batch["attention_mask"].shape}')
        break

    print('finished')

