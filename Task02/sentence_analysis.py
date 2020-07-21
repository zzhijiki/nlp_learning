import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class SentenceAnalysis:
    def __init__(self, data_path, n_classes=None, with_label=True):
        self.data_path = data_path
        self.with_label = with_label  # 测试集无标签导入
        self.n_classes = n_classes
        self.load_dataset()

    @property
    def data(self):
        if self.with_label:
            return self.X, self.Y
        else:
            return self.X

    def load_dataset(self):
        if self.with_label:
            train = pd.read_csv(self.data_path, sep='\t')
            self.X = train[[col for col in train.columns if col != "label"]]
            self.Y = train["label"]
        else:
            test = pd.read_csv(self.data_path)
            self.X = test
            self.Y = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        """Generate one  of data"""
        x = self.X.iloc[int(index)]
        if self.with_label:
            y = self.Y[int(index)]
            # y=one_hot(y,self.n_classes)
            return x, y
        else:
            return x

    def passage_length_ana(self, show_describe=True, show_hist=False):
        """
        句子长度分析
        """
        df = self.X.copy()
        df["text_len"] = df.text.apply(lambda x: len(x.split(" ")))
        if show_describe:
            print(df["text_len"].describe())
        if show_hist:
            df.text_len.hist(bins=100)
            plt.xlabel('Text char count')
            plt.title("Histogram of char count");
        return df["text_len"]

    def show_hist(self, data, bins=100, title="Not define.", xlabel="no xlabel."):
        data.hist(bins=bins)
        plt.xlabel(xlabel)
        plt.title(title);
        return

    def label_distribution(self, show_bar=True, title='class count', xlabel="category"):
        """
        label分布的分析
        """
        if not self.with_label:
            print("没有可用的标签！")
            return
        df = self.X.copy()
        df["label"] = self.Y.values
        df_label = df.groupby("label").agg({"text": ["count"]})
        if show_bar:
            df["label"].value_counts().plot(kind="bar")
            plt.title(title)
            plt.xlabel(xlabel);
        return df_label

    def word_distribution(self, show_most=1, show_least=1):
        """
        字符分布
        """
        show_most, show_least = int(show_most), int(show_least)
        df = self.X.copy()
        all_lines = " ".join(list(df["text"]))
        word_count = Counter(all_lines.split(" "))
        if show_most > 0:
            print("最多的{}个字符:".format(show_most))
            print(word_count.most_common(int(show_most)))
        if show_least > 0:
            print("最少的{}个字符:".format(show_least))
            print(word_count.most_common()[-int(show_least):])
        print("所有文档中拥有字符数： {}".format(len(word_count)))
        return word_count

    def word_in_sentece_distribution(self, show_most=1, show_least=0):
        """
        统计了不同字符在句子中出现的次数
        """
        show_most, show_least = int(show_most), int(show_least)
        df = self.X.copy()
        df['text_unique'] = df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
        all_lines = ' '.join(list(df['text_unique']))
        word_count = Counter(all_lines.split(" "))
        if show_most > 0:
            print("最多的{}个字符:".format(show_most))
            for k, v in word_count.most_common(show_most):
                print("字符编号为 {:>4} 在所有句子中的比例为: {:.2%}".format(k, v / self.X.shape[0]))
        if show_least > 0:
            print("最少的{}个字符:".format(show_least))
            for k, v in word_count.most_common()[-int(show_least):]:
                print("字符编号为 {:>4} 在所有句子中的比例为: {:.2%}".format(k, v / self.X.shape[0]))
        return word_count

    def word_groupbylabel_count(self, show_most=1):
        """
        统计每类新闻中出现次数最多的字符
        """
        show_most = int(show_most)
        if not self.with_label:
            print("没有可用的标签！")
            return
        df = self.X.copy()
        df["label"] = self.Y.values
        word_group_count = {}
        for name, group in df[["label", "text"]].groupby("label"):
            all_lines = " ".join(list(group.text))
            word_count = Counter(all_lines.split(" "))
            word_group_count[name] = word_count
        if show_most > 0:
            if not self.n_classes:
                self.n_classes = self.Y.nunique()
            for i in range(self.n_classes):
                print("标签为第{:>2d}组，最多的{}个单词为 {} ".format(i, show_most, word_group_count[i].most_common(show_most)))
        return word_group_count

    def last_word_ana(self, show_most=1, show_least=1):
        """
        句尾分析
        """
        show_most, show_least = int(show_most), int(show_least)
        df = self.X.copy()
        df["last_word"] = df.text.apply(lambda x: x.split(" ")[-1])
        last_word_count = Counter(df["last_word"])
        if show_most > 0:
            print("最多的{}个字符:".format(show_most))
            print(last_word_count.most_common(int(show_most)))
        if show_least > 0:
            print("最少的{}个字符:".format(show_least))
            print(last_word_count.most_common()[-int(show_least):])
        print("所有文档中不同的最后一个字符数： {}".format(len(last_word_count)))
        return last_word_count
