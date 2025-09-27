import torch


def evaluate(model, test_loader,device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(inputs,attention_mask)
            predict = torch.argmax(logits, dim=1)
            correct += (predict == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total
    print('Accuracy:', accuracy)
    return accuracy