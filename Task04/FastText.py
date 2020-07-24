import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, vocab, embedding_dim=300, hidden_dim=64, out_dim=14):
        super(FastText, self).__init__()

        self.num_vocab = len(vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.embed = nn.Embedding(num_embeddings=self.num_vocab, embedding_dim=self.embedding_dim)
        self.embed.weight.requires_grad = True

        self.pred = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.Dropout(0.5),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
            #             nn.ReLU(inplace=True),
        )

        nn.init.xavier_uniform_(self.embed.weight)
        self.xavier(self.pred)

    def xavier(self, layers):
        for index, net in enumerate(layers):
            if index == 0:
                nn.init.xavier_uniform_(net.weight)

    def forward(self, x):
        x = self.embed(x)
        out = self.pred(torch.mean(x, dim=1))
        return out
