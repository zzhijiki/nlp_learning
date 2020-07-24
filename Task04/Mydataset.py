import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator


class MyDataset:
    def __init__(self, train_path='./data/train_torch.csv', test_path='./data/test_a.csv', fix_length=600):
        self.train_path = train_path
        self.test_path = test_path
        self.fix_length = fix_length

    def get_data_by_torchtext(self):
        print("读取数据，需要花挺长时间")

        def x_tokenize(x):
            return [w for w in x.split()]

        def y_tokenize(y):
            return int(y)

        TEXT = Field(sequential=True, tokenize=x_tokenize,
                     fix_length=self.fix_length, use_vocab=True,
                     init_token=None, eos_token=None,
                     include_lengths=True, batch_first=True)

        LABEL = Field(sequential=False, tokenize=y_tokenize,
                      use_vocab=False, is_target=True)

        fields_train = [('text', TEXT), ('label', LABEL)]
        fields_test = [('text', TEXT)]

        train = TabularDataset(
            path=self.train_path, format='csv',
            skip_header=True, fields=fields_train
        )
        test = TabularDataset(
            path=self.test_path, format='csv',
            skip_header=True, fields=fields_test
        )

        TEXT.build_vocab(train)
        return TEXT.vocab, train, test

    def split(self, train, split_ratio=0.9):
        train_dataset, valid_dataset = train.split(split_ratio, stratified=True)
        return train_dataset, valid_dataset

    def get_iter(self, train, valid, test, train_batch=64, valid_batch=64, test_batch=128):
        # 构造迭代器
        train_iter, valid_iter = BucketIterator.splits(
            (train, valid),
            batch_sizes=(train_batch, valid_batch),
            device=torch.device("cuda"),
            sort_key=lambda x: len(x.text),
            sort_within_batch=True
        )

        test_iter = Iterator(test, batch_size=test_batch, device=torch.device("cuda"), sort=False, repeat=False,
                             sort_within_batch=False, shuffle=False)
        return train_iter, valid_iter, test_iter
