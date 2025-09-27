import sys

sys.path.append("../")

import torch
from src.data.dataloader import getDataloader
from src.model.bert_classifier import BertSpamClassifier
from src.utils.tokenizer_utilis import get_tokenizer
from src.train.trainer import train_model
from src.train.evaluator import evaluate
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

BERT_PATH = '/home/wby/projects/model/bert-base-cased'
CSV_PATH = '/home/wby/projects/week4/data/final.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
tokenize = get_tokenizer(BERT_PATH)

def main():
    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('/home/wby/projects/week4/data/train.csv', index=False)
    test_df.to_csv('/home/wby/projects/week4/data/test.csv', index=False)

    train_dataloader = getDataloader(
        '/home/wby/projects/week4/data/train.csv',
        tokenizer=tokenize,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_dataloader = getDataloader(
        '/home/wby/projects/week4/data/test.csv',
        tokenizer=tokenize,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BertSpamClassifier(BERT_PATH)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model = train_model(
        model,
        train_dataloader,
        DEVICE,
        criterion,
        optimizer,
        EPOCHS,
    )

    evaluate(model, test_dataloader, DEVICE)
    print('模型完成')

if __name__ == '__main__':
    main()