import numpy as np
import torch
from transformers import BertTokenizer,BertModel
import torch.nn as nn
from tqdm import tqdm
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention


#模拟神经网络的隐藏层
#其内部只有初始化和前向传导
class BertSpamClassifier(nn.Module):
    #num_labels代表最终输出维数
    def __init__(self, bert_path, num_labels=2):
        super(BertSpamClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size,num_labels)#线性映射，将当前维数据映射到2维

    #前向传导可以理解为自己去写一个model.fit(train_X,train_y)
    def forward(self,input_ids,attention_mask):
        #把outputs理解为通过bert模型拿到了这一块Dataloader的向量结果报告
        #(32,768)因为bert的特性会得到768维度的特征维度
        _,outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # 不返回字典，返回元组
        )
        x = self.dropout(outputs) #去掉768维的结果报告里10%的依赖数据，防止过拟合
        logits = self.fc(x) # 利用类中定义的线性映射函数fc，来得到2维结果向量
        return logits #返回2维结果，让下一个bash的DataLoader进行训练参照
        #logits得到的是一个(32,2)的结果数组集，内部是原始分数