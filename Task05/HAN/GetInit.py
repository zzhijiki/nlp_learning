import pandas as pd


class GetInit:
    def __init__(self, data_root):
        print("GetInit Start!")
        self.data_root = data_root
        self.x_train, self.y_train, self.x_test = self.get_pandas()
        print("GetInit End!")

    def get_pandas(self):
        train = pd.read_csv(self.data_root["train_path"])
        train["corpus"] = train.text.apply(lambda x: x.split(" "))
        test = pd.read_csv(self.data_root["test_path"])
        test["corpus"] = test.text.apply(lambda x: x.split(" "))
        x_train = train.corpus.values
        y_train = train.label.values
        x_test = test.corpus.values
        del train
        del test
        return x_train, y_train, x_test
