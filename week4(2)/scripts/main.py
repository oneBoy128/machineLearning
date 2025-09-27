import sys
from itertools import chain

import pandas as pd

print("脚本开始运行！")  # 加这行测试

import torch
from transformers import BertTokenizer,BertModel
from sklearn.model_selection import train_test_split


# 把项目根目录加入Python路径（确保能导入src下的模块）
sys.path.append("../")
from src.data.dataloader import getDataloader              # 导入数据加载
from src.train.trainer import train_model
from src.train.evaluator import evaluate
from src.model.bert_classifier import BertSpamClassifier


CSV_PATH = '/home/wby/projects/week4(2)/data/final.csv'
BERT_PATH = '/home/wby/projects/model/bert-base-cased'
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

def main():
    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('/home/wby/projects/week4(2)/data/train_df.csv', index=False)
    test_df.to_csv('/home/wby/projects/week4(2)/data/test_df.csv', index=False)

    train_loader = getDataloader(
        '/home/wby/projects/week4(2)/data/train_df.csv',
        tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = getDataloader(
        '/home/wby/projects/week4(2)/data/test_df.csv',
        tokenizer,
        shuffle=False
    )

    model = BertSpamClassifier(BERT_PATH)
    """
    旧代码，统一指定所有隐藏层参数的学习率为2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    """

    optimizer_grouped_parameters = [
        # BERT底层（前8层）
        {"params": model.bert.encoder.layer[:8].parameters(), "lr": 1.6e-5, "weight_decay": 0.01},
        # BERT高层（后4层）
        {"params": model.bert.encoder.layer[8:].parameters(), "lr": 2e-5, "weight_decay": 0.01},
        # 自定义分类头（正确合并参数）
        {"params": chain(model.fc.parameters(), model.dropout.parameters()), "lr": 2e-5, "weight_decay": 0.01}
    ]

    # 3. 优化器（AdamW，论文常用）
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999))

    criterion = torch.nn.CrossEntropyLoss()

    model = train_model(
        model=model,
        device=DEVICE,
        train_dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS
    )

    evaluate(model, test_loader, DEVICE)
    print('finish')


if __name__ == "__main__":
    main()



