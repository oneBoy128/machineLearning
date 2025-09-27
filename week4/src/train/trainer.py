import torch
from tqdm import tqdm

def train_model(model,device,train_loader,criterion,optimizer,epochs=3):
    model.train()
    model.to(device)
    for i in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader,desc=f'Training{i}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {i+1}/{epochs}\tLoss: {avg_loss:.3f}')
    return model
