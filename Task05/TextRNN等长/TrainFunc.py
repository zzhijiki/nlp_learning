from copy import deepcopy
import torch
from sklearn.metrics import f1_score


class TrainFunc:
    def __init__(self, model, criterion, opt, train_iter=None, valid_iter=None, test_iter=None):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.best_model = model
        self.best_score = 0
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter

    def train(self, epoch):
        self.model.train()

        for i in range(epoch):
            train_acc = 0
            train_loss = 0
            for batch_idx, (data, _, label) in enumerate(iter(self.train_iter)):
                data = data.cuda()
                label = label.cuda()
                batchsize = data.shape[0]
                output = self.model(data)
                self.opt.zero_grad()
                loss = self.criterion(output, label)
                loss.backward()
                self.opt.step()
                train_loss += loss.item()
                train_acc += (output.argmax(1) == label).sum().item()
                if batch_idx % int(200 * (64 / batchsize)) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i + 1, batch_idx * len(data), len(self.train_iter.dataset),
                        100. * batch_idx / len(self.train_iter), loss.item()))
            print(
                f'\tLoss: {train_loss / len(self.train_iter):.4f}(train)\t|\tAcc: {train_acc / len(self.train_iter.dataset) * 100:.1f}%(train)')

            score = self.valid_func()
            if score > self.best_score:
                self.best_score = score
                self.best_model = deepcopy(self.model)
                print("Now_best:{:.4f}".format(self.best_score))
        #         scheduler.step()
        return self.best_model

    def valid_func(self):
        valid_acc = 0
        valid_loss = 0
        ans_box = []
        label_box = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, _, label) in enumerate(iter(self.valid_iter)):
                data = data.cuda()
                label = label.cuda()
                # batchsize = data.shape[0]

                output = self.model(data)
                pred = output.argmax(1)

                loss = self.criterion(output, label)

                ans_box.extend(pred.cpu().tolist())
                label_box.extend(label.cpu().tolist())
                valid_loss += loss.item()
                valid_acc += (pred == label).sum().item()

            score1 = f1_score(ans_box, label_box, average='macro')
            score2 = f1_score(ans_box, label_box, average='micro')
            print(
                f'\tLoss: {valid_loss / len(self.valid_iter):.4f}(valid)\t|\tAcc: {valid_acc / len(self.valid_iter.dataset) * 100:.1f}%(valid)')
            print(f'\tMicro: {score2:.4f}(valid)\t|\tMacro: {score1:.4f}(valid)')

        self.model.train()
        return score1

    def predict(self):
        self.best_model.eval()
        ans_box = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(iter(self.test_iter)):
                data = data.cuda()

                output = self.best_model(data)
                pred = output.argmax(1)

                ans_box.extend(pred.cpu().tolist())
        return ans_box
