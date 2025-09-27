
from ..data.dataloader import getdataloader
from .tokenizer_utilis import get_tokenizer

BERT_PATH = '/home/wby/projects/model/bert-base-cased'
CSV_PATH = '/home/wby/projects/week4/data/final.csv'

def test_bert():
    tokenizer = get_tokenizer(BERT_PATH)
    dataloader = getdataloader(
        CSV_PATH,
        tokenizer,
        batch_size=8,
        shuffle=False
    )

    for batch in dataloader:
        print(batch['input_ids'].shape) #8,128
        print(batch['attention_mask'].shape)  #8,128
        break

    print('finished')


if __name__ == '__main__':
    test_bert()