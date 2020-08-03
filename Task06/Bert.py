import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, bert_path, hidden_size=128, output_size=14, dropout=0.5):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(
            bert_path,
            num_labels=14,
            output_attentions=False,  # 模型是否返回 attentions weights.
            output_hidden_states=False,  # 模型是否返回所有隐层状态.
        )

        self.fc1 = nn.Linear(258, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, data, mask, feature):
        _, out = self.bert(data, token_type_ids=None, attention_mask=mask)
        out = torch.cat((out, feature), dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        return F.log_softmax(out, 1)
