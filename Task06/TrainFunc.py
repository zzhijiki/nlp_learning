from copy import deepcopy
import torch
from sklearn.metrics import f1_score
import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class TrainFunc:
    def __init__(self, model, criterion, opt, schedule, train_iter=None, valid_iter=None, test_iter=None):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.schedule = schedule
        self.best_model = model
        self.best_score = 0
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.training_stats = []

    def train(self, epoch):

        total_t0 = time.time()

        for epoch_i in range(0, epoch):
            print(" ")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epoch))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            train_acc = 0
            # 训练集小批量迭代
            for step, (data, mask, feature, label) in enumerate(iter(self.train_iter)):
                batch_size = data.shape[0]
                data = data.cuda()
                mask = mask.cuda()
                feature = feature.cuda()
                label = label.cuda()

                self.opt.zero_grad()
                output = self.model(data, mask, feature)
                loss = self.criterion(output, label)
                loss.backward()
                total_train_loss += loss.item()
                train_acc += (output.argmax(1) == label).sum().item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.schedule.step()
                if step % int(80 * (8 / batch_size)) == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.  Loss:{:<20,}   Elapsed: {:}.'.format(step,
                                                                                             len(self.train_iter),
                                                                                             loss.item(), elapsed))
            # 平均训练误差
            avg_train_loss = total_train_loss / len(self.train_iter)
            # 单次 epoch 的训练时长
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            print("  Training acc: {0:.4f}".format(train_acc / len(self.train_iter.dataset) * 100))
            score, avg_val_loss, avg_val_accuracy, validation_time = self.valid_func()
            if score > self.best_score:
                self.best_score = score
                self.best_model = deepcopy(self.model)
                print("  Now_best:{:.4f}".format(self.best_score))
            #         scheduler.step()
            self.training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Acc.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
        return self.best_model

    def valid_func(self):
        print("")
        print("Running Validation...")
        t0 = time.time()
        self.model.eval()
        valid_acc = 0
        valid_loss = 0
        nb_eval_steps = 0
        ans_box = []
        label_box = []
        for batch_idx, (data, mask, feature, label) in enumerate(iter(self.valid_iter)):
            batch_size = data.shape[0]
            data = data.cuda()
            mask = mask.cuda()
            feature = feature.cuda()
            label = label.cuda()

            with torch.no_grad():
                output = self.model(data, mask, feature)
                loss = self.criterion(output, label)
            pred = output.argmax(1)
            valid_loss += loss.item()
            valid_acc += (pred == label).sum().item()

            ans_box.extend(pred.cpu().tolist())
            label_box.extend(label.cpu().tolist())
        score1 = f1_score(ans_box, label_box, average='macro')
        score2 = f1_score(ans_box, label_box, average='micro')

        avg_val_accuracy = valid_acc / len(self.valid_iter.dataset) * 100
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = valid_loss / len(self.valid_iter)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        print("  Micro score: {:}".format(score2))
        print("  Macro score: {:}".format(score1))
        # 记录本次 epoch 的所有统计信息

        return score1, avg_val_loss, avg_val_accuracy, validation_time

    def predict(self):
        self.best_model.eval()
        t0 = time.time()
        ans_box = []
        with torch.no_grad():
            for step, (data, mask, feature) in enumerate(iter(self.test_iter)):
                if step % int(40) == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.  Elapsed: {:}.'.format(step, len(self.test_iter), elapsed))
                data = data.cuda()
                mask = mask.cuda()
                feature = feature.cuda()
                with torch.no_grad():
                    output = self.best_model(data, mask, feature)
                pred = output.argmax(1)
                ans_box.extend(pred.cpu().tolist())
        return ans_box
