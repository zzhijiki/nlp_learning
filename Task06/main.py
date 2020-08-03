import torch
from torch import nn

from .GetLoader import GetLoader
from .MyDataset import MyDataset
from .GetInit import GetInit
from .TrainFunc import TrainFunc
from .Bert import Bert
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np

if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    data_root = {
        "train_path": './data/train_torch.csv',
        "test_path": "./data/test_a.csv",
        "sub_path": "./data/test_a_sample_submit.csv",
        # "w2v_path": "./data/word2vec.bin",
        "bert_path": "./bert-mini/"
    }
    config = GetInit(data_root)

    train_dataset = MyDataset(data_root["bert_path"],
                              corpus=config.x_train,
                              feature=config.x_train_feature,
                              corpus_label=config.y_train,
                              with_label=True)
    test_dataset = MyDataset(data_root["bert_path"],
                             corpus=config.x_test,
                             feature=config.x_test_feature,
                             with_label=False)

    loader = GetLoader(train_dataset, test_dataset)

    # 建立model
    model = Bert(bert_path=data_root["bert_path"], hidden_size=100, output_size=14, dropout=0.5)
    model.cuda()
    criterion = nn.NLLLoss()
    opt = AdamW(model.parameters(),
                lr=2e-5,  # args.learning_rate - default is 5e-5
                eps=1e-8  # args.adam_epsilon  - default is 1e-8
                )
    epochs = 2
    total_steps = len(loader.train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(opt,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # 开始训练
    mytrain = TrainFunc(model, criterion, opt, scheduler, loader.train_loader, loader.valid_loader, loader.test_loader)
    best_model = mytrain.train(epochs)
