import torch
from tqdm import tqdm


def train_model(model,train_dataloader,device,criterion,optimizer,epochs=3):
    model.to(device) #送入cupd设备里
    model.train()
    for i in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch{epochs+1}/{epochs}"):
            #把这些数据转到GPU里面
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            #前向传导，某一批数据训练完毕（32行128列）
            optimizer.zero_grad() #清理梯度
            logits = model(input_ids,attention_mask)
            loss = criterion(logits,label) #注意，这里是交叉熵损失，不是最后的训练集准度评估

            #开始梯度算法更新参数
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {i+1}/{epochs}\tLoss: {avg_loss}")

    return model

