import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, bert_path, corpus, feature, corpus_label=None, max_length=256, with_label=False):
        super(MyDataset, self).__init__()
        self.corpus = corpus
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.with_label = with_label
        self.max_length = max_length
        self.feature = feature

        if self.with_label:
            self.corpus_label = torch.tensor(corpus_label)

    def __getitem__(self, item):
        encoded_dict = self.tokenizer.encode_plus(
            self.corpus[item],  # 输入文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=self.max_length,  # 填充 & 截断长度
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        if self.with_label:
            return encoded_dict['input_ids'].squeeze(0), encoded_dict['attention_mask'].squeeze(0), \
                   torch.FloatTensor(self.feature[item]), self.corpus_label[item]
        else:
            return encoded_dict['input_ids'].squeeze(0), encoded_dict['attention_mask'].squeeze(0), \
                   torch.FloatTensor(self.feature[item])

    def __len__(self):
        return len(self.corpus)
