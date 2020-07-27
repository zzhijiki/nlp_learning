import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=100, output_size=14, dropout=0.5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.word_lstm = nn.LSTM(input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def init_weights(self, pre_trained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pre_trained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False
        else:
            self.embedding.weight.requires_grad = True

    # def forward(self, inputs):
    #     out = self.embedding(inputs)
    #     out, _ = self.word_lstm(out)
    #     out = self.dropout(out)
    #     out = self.fc(out[:, -1, :])
    #     return F.log_softmax(out, 1)

    def forward(self, x):
        _pad, _len = pad_packed_sequence(x, batch_first=True)
        x_embedding = self.embedding(_pad)
        x_embedding = x_embedding.view(x_embedding.shape[0], x_embedding.shape[1], -1)
        output, _ = self.word_lstm(x_embedding)
        temp = []
        _len = _len - 1
        for i in range(len(_len)):
            temp.append(output[i, _len[i], :].tolist())
        temp = torch.tensor(temp).cuda()
        # print(temp.shape)
        out = self.fc(temp)
        # print(out.shape)
        return F.log_softmax(out, 1)
