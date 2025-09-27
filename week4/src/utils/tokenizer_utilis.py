from transformers import BertTokenizer

def get_tokenizer(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    return tokenizer