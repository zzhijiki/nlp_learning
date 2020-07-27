import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(2, 3, 4), dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        '''
        上面是个for循环，不好理解写成下面也是没问题的。
        self.conv13 = nn.Conv2d(Ci, Co, (2, D))
        self.conv14 = nn.Conv2d(Ci, Co, (3, D))
        self.conv15 = nn.Conv2d(Ci, Co, (4, D))
        '''
        # kernal_size = (K,D)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False
        else:
            self.embedding.weight.requires_grad = True

    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs).unsqueeze(1)  # (B,1,T,D)
        # print(inputs.shape)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # print(inputs[0].shape)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        '''
        最大池化也可以拆分理解
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        concated = torch.cat(inputs, 1)
        if is_training:
            concated = self.dropout(concated)  # (N,len(Ks)*Co)
        out = self.fc(concated)
        return F.log_softmax(out, 1)
