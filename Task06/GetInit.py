import pandas as pd
import re
import numpy as np


class GetInit:
    def __init__(self, data_root):
        print("GetInit Start!")
        self.data_root = data_root
        self.x_train, self.y_train, self.x_test, self.x_train_feature, self.x_test_feature = self.get_pandas()
        print("GetInit End!")

    def get_pandas(self):
        train = pd.read_csv(self.data_root["train_path"])
        test = pd.read_csv(self.data_root["test_path"])

        x_train = train.text.str.replace("900", "[SEP]").replace("3750", "[SEP]").values
        y_train = train.label.values
        x_test = test.text.str.replace("900", "[SEP]").replace("3750", "[SEP]").values

        train["length"] = train.text.apply(lambda x: len(x.split(" ")))
        test["length"] = test.text.apply(lambda x: len(x.split(" ")))
        #         train["length"]=(train["length"]-train["length"].min())/(train["length"].max()-train["length"].min())
        #         test["length"]=(test["length"]-test["length"].min())/(test["length"].max()-test["length"].min())
        train["length"] = np.log10(train["length"]) / np.log10(train["length"].max())
        test["length"] = np.log10(test["length"]) / np.log10(test["length"].max())

        train["sentence_length"] = train.text.apply(lambda x: len(re.split(" 3750 | 900 ", x)))
        test["sentence_length"] = test.text.apply(lambda x: len(re.split(" 3750 | 900 ", x)))
        #         train["sentence_length"]=(train["sentence_length"]-train["sentence_length"].min())/(train["sentence_length"].max()-train["sentence_length"].min())
        #         test["sentence_length"]=(test["sentence_length"]-test["sentence_length"].min())/(test["sentence_length"].max()-test["sentence_length"].min())
        train["sentence_length"] = np.log10(train["sentence_length"]) / np.log10(train["sentence_length"].max())
        test["sentence_length"] = np.log10(test["sentence_length"]) / np.log10(test["sentence_length"].max())

        x_train_feature = train[["length", "sentence_length"]].values
        x_test_feature = test[["length", "sentence_length"]].values

        del train
        del test
        return x_train, y_train, x_test, x_train_feature, x_test_feature
