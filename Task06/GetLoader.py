import torch
from torch.utils.data import DataLoader


class GetLoader:
    def __init__(self, train_dataset, test_dataset, split_ratio=0.9):
        self.ratio = split_ratio
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataset, self.valid_dataset = self.split()

        self.train_loader, self.valid_loader, self.test_loader = None, None, None
        self.get_iter()
        print("GetLoader End")

    def split(self):
        train_size = int(self.ratio * len(self.train_dataset))
        valid_size = len(self.train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, valid_size])
        return train_dataset, valid_dataset

    def get_iter(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=128)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128)
