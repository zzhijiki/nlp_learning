import torch
from torch.utils.data import Dataset
import re

class MyDataset(Dataset):
    def __init__(self, model_embedding, corpus, corpus_label=None, max_length=40, with_label=False):
        super(MyDataset, self).__init__()
        self.corpus = corpus
        self.with_label = with_label
        if self.with_label:
            self.corpus_label = torch.tensor(corpus_label)

        self.max_length = max_length
        self.model_embedding = model_embedding
        self.word_dict = self.model_embedding.word2id

    def __getitem__(self, item):
        m=re.split(" 3750 | 900 | 648 "," ".join(self.corpus[item]))
        m=map(lambda x:x.split(" "),m)
        m=list(m)
        k=[x for x in m if len(x)>3]
        if len(k)!=0:m=k
        document=[]
        for sentence in m:
            temp=[]
            for word in sentence:
                if word in self.model_embedding.word2id:
                    temp.append(self.model_embedding.word2id[word])
                else:
                    temp.append(self.model_embedding.word2id['<UNKNOWN>'])
                    
            if len(temp) > self.max_length:
                temp = temp[:self.max_length]
            while len(temp) <self.max_length:
                temp.append(self.model_embedding.word2id['<PADDING>'])
                
            document.append(torch.tensor(temp))
        if len(document) > self.max_length:
            document = document[:self.max_length]
        if self.with_label:
            return torch.stack(document), self.corpus_label[item]
        else:
            return torch.stack(document)          

    def __len__(self):
        return len(self.corpus)
