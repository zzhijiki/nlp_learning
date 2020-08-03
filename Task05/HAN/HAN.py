import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, X):
        out = self.layer1(X)
        out = torch.tanh(out)
        out = self.layer2(out)
        out = F.softmax(out, dim=-2)
        return out.mul(X).sum(-2)


class HAN_Attention_revise(nn.Module):
    """
    HAN的attention更新,dot做法
    sentence_level:input_size =b*l*300,outsize=b*300
    """

    def __init__(self, in_size=300, hidden_size=200):
        super(HAN_Attention_revise, self).__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.us = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)  # 200

    def forward(self, X):
        X1 = X.view(X.shape[1], X.shape[0], X.shape[2])  # l前置， l*b*300
        U = torch.tanh(self.layer1(X1))  # 构造U l*b*200
        A = (U * self.us).sum(2)  # 点积，维度 U*self.us  = l*b*200
        A = F.softmax(A.T, 1).unsqueeze(2)  # softmax 操作+转置，  b*l*1  第1维归一化
        return A.mul(X).sum(1)


class HAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=100, output_size=14, dropout=0.5):
        super(HAN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.att1 = Attention(hidden_size * 2, 128)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size // 2, bidirectional=True, batch_first=True)
        self.att2 = Attention(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False
        else:
            self.embedding.weight.requires_grad = True

    def forward(self, X):
        X = self.embedding(X)
        out = X.view(X.shape[0] * X.shape[1], X.shape[2], -1).contiguous()
        out, _ = self.gru1(out)
        out = out.view(X.shape[0], X.shape[1], X.shape[2], -1).contiguous()
        out = self.att1(out)
        out, _ = self.gru2(out)
        out = self.att2(out)
        out = self.dropout(out)
        out = self.fc(out)
        return F.log_softmax(out, 1)
