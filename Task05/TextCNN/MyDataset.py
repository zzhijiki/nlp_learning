import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, model_embedding, corpus, corpus_label=None, max_length=600, with_label=False):
        super(MyDataset, self).__init__()
        self.corpus = corpus
        self.with_label = with_label
        if self.with_label:
            self.corpus_label = torch.tensor(corpus_label)

        self.max_length = max_length
        self.model_embedding = model_embedding
        self.word_dict = self.model_embedding.word2id

    def __getitem__(self, item):
        temp = []
        for word in self.corpus[item]:
            if word in self.model_embedding.word2id:
                temp.append(self.model_embedding.word2id[word])
            else:
                temp.append(self.model_embedding.word2id['<UNKNOWN>'])
        if len(temp) > self.max_length:
            temp = temp[:self.max_length]
        if self.with_label:
            return torch.tensor(temp), self.corpus_label[item]
        else:
            return torch.tensor(temp)

    def __len__(self):
        return len(self.corpus)
