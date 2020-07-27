import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class GetLoader:
    def __init__(self, train_dataset, test_dataset, split_ratio=0.9):
        self.ratio = split_ratio
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataset, self.valid_dataset = self.split()

        self.train_loader, self.valid_loader, self.test_loader = None,None,None
        self.get_iter()
        print("GetLoader End")

    def split(self):
        train_size = int(self.ratio * len(self.train_dataset))
        valid_size = len(self.train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, valid_size])
        return train_dataset, valid_dataset

    def get_iter(self):
        def collate_fn_train(batch_data):
            batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
            data_length = [len(xi[0]) for xi in batch_data]
            sent_seq = [xi[0] for xi in batch_data]
            label = [xi[1] for xi in batch_data]
            padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
            return padded_sent_seq, data_length, torch.tensor(label, dtype=torch.long)

        def collate_fn_test(batch_data):
            data_length = [len(xi) for xi in batch_data]
            sent_seq = [xi for xi in batch_data]
            padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
            return padded_sent_seq, data_length

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, collate_fn=collate_fn_train)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=256, collate_fn=collate_fn_train)
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, collate_fn=collate_fn_test)
