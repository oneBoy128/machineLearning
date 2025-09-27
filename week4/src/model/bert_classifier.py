import torch
from torch import nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self,bert_path, num_labels=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, attention_mask):
        _,outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.dropout(outputs)
        logits = self.fc(x)
        return logits