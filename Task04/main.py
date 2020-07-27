from FastText import FastText
from Mydataset import MyDataset
from TranFunc import TrainFunc
import torch
import torch.nn as nn
from FocalLoss import FocalLoss

if __name__ == "__main__":
    my_data = MyDataset(train_path='./data/train_torch.csv', test_path='./data/test_a.csv', fix_length=600)
    vocab, train, test = my_data.get_data_by_torchtext()
    train_dataset, valid_dataset = my_data.split(train)
    train_iter, valid_iter, test_iter = my_data.get_iter(train_dataset, valid_dataset, test, train_batch=64)
    # 构造模型的参数
    model = FastText(vocab)
    model = model.cuda()
    # criterion = FocalLoss(14).cuda()
    criterion = nn.NLLLoss()
    lr = 1e-4
    opt = torch.optim.Adam(model.parameters(), lr)
    # 开始训练，使用验证集获得做好的参数，并进行预测
    mytrain = TrainFunc(model, criterion, opt, train_iter, valid_iter, test_iter)
    best_model = mytrain.train(15)
    ans = mytrain.predict()  # 因为没有打乱过，所以这个ans是按顺序的。

