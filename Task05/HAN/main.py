import torch
from torch import nn

from GetLoader import GetLoader
from ModelEmbedding import ModelEmbedding
from MyDataset import MyDataset
from GetInit import GetInit
from TrainFunc import TrainFunc
from TextRNN import TextRNN

if __name__ == "__main__":
    data_root = {
        "train_path": '../../data/train_torch.csv',
        "test_path": "../../data/test_a.csv",
        "sub_path": "../../data/test_a_sample_submit.csv",
        "w2v_path": "../../data/word2vec.bin"
    }
    config = GetInit(data_root)
    model_embedding = ModelEmbedding(data_root["w2v_path"])

    train_dataset = MyDataset(model_embedding,
                            corpus=config.x_train,
                            corpus_label=config.y_train,
                            with_label=True,
                            max_length=40)
    test_dataset = MyDataset(model_embedding,
                            corpus=config.x_test,
                            with_label=False,
                            max_length=40)

    loader = GetLoader(train_dataset, test_dataset)

    # 建立model
    model=HAN(model_embedding.dict_length, embedding_dim=300,hidden_size=100, output_size=14, dropout=0.5)
    model.init_weights(model_embedding.embedding, is_static=False)
    model = model.cuda()
    criterion = nn.NLLLoss()
    # opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    # opt = torch.optim.Adagrad(model.parameters(), lr=1e-2)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    # opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler  = lr_scheduler.CosineAnnealingLR(opt, T_max=5,eta_min=4e-08)

    # 开始训练
    mytrain = TrainFunc(model, criterion, opt, scheduler, loader.train_loader, loader.valid_loader, loader.test_loader)
    best_model = mytrain.train(25)
