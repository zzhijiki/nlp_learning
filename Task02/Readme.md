## 数据集样式：

基于 【天池训练赛：零基础入门NLP之新闻文本分类】 的数据构建一个句子分析的类，用来进行数据分析。

已经上传至[Github](https://github.com/zzhijiki/nlp_learning/tree/master/Task02)


训练集：要有label
| label |                             text                             |
| :---: | :----------------------------------------------------------: |
|   6   | 57 44 66 56 2 3 3 37 5 41 9 55  |
测试集：没有label
| index |                             text                             |
| :---: | :----------------------------------------------------------: |
|   1   | 57 44 66 56 2 3 3 37 5 41 9 55  |

构建一个句子分析的类。

```python
import numpy as np
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


```

#### 功能展示：
- 对于训练集：
```python
train_path="../data/train_set.csv"
sentence_train=SentenceAnalysis(train_path,n_classes=14,with_label=True)
```
```python
# 功能展示
# __getitem__
sentence_train[1]
# output
(text    4464 486 6352 5619 2465 4802 1452 3137 5778 54...
 Name: 1, dtype: object,
 11)

# __len__
len(sentence_train)
# output
200000


# data
train_X,train_y=sentence_train.data
# output
略，train_X是一个DataFrame


# 文章长度分析
df_length=sentence_train.passage_length_ana()
# output
count    200000.000000
mean        907.207110
std         996.029036
min           2.000000
25%         374.000000
50%         676.000000
75%        1131.000000
max       57921.000000
Name: text_len, dtype: float64


# 辅助的作图
sentence_train.show_hist(df_length,100,'Text char count',"Histogram of char count")
# output
略
```

```python
# 新闻类别分布
df_label=sentence_train.label_distribution()
# output
```
![](https://upload-images.jianshu.io/upload_images/24165403-83d2c90f339e4cbf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
# 字符个数分布
word_dict=sentence_train.word_distribution(5,5)
# output
最多的5个字符:
[('3750', 7482224), ('648', 4924890), ('900', 3262544), ('3370', 2020958), ('6122', 1602363)]
最少的5个字符:
[('155', 1), ('1415', 1), ('1015', 1), ('4468', 1), ('3133', 1)]
所有文档中拥有字符数： 6869


# 不同字符在句子中出现的次数
word_in_sentece_dict=sentence_train.word_in_sentece_distribution(5)
# output
最多的5个字符:
字符编号为 3750 在所有句子中的比例为: 99.00%
字符编号为  900 在所有句子中的比例为: 98.83%
字符编号为  648 在所有句子中的比例为: 95.99%
字符编号为 2465 在所有句子中的比例为: 88.66%
字符编号为 6122 在所有句子中的比例为: 88.27%


# 统计每类标签中出现次数最多的字符
word_group_count=sentence_train.word_groupbylabel_count(5)
# output 
标签为第 0组，最多的5个单词为 [('3750', 1267331), ('648', 967653), ('900', 577742), ('3370', 503768), ('4464', 307431)] 
标签为第 1组，最多的5个单词为 [('3750', 1200686), ('648', 714152), ('3370', 626708), ('900', 542884), ('4464', 445525)] 
标签为第 2组，最多的5个单词为 [('3750', 1458331), ('648', 974639), ('900', 618294), ('7399', 351894), ('6122', 343850)] 
标签为第 3组，最多的5个单词为 [('3750', 774668), ('648', 494477), ('900', 298663), ('6122', 187933), ('4939', 173606)] 
标签为第 4组，最多的5个单词为 [('3750', 360839), ('648', 231863), ('900', 190842), ('4411', 120442), ('7399', 86190)] 
标签为第 5组，最多的5个单词为 [('3750', 715740), ('648', 329051), ('900', 305241), ('6122', 159125), ('5598', 136713)] 
标签为第 6组，最多的5个单词为 [('3750', 469540), ('648', 345372), ('900', 222488), ('6248', 193757), ('2555', 175234)] 
标签为第 7组，最多的5个单词为 [('3750', 428638), ('648', 262220), ('900', 184131), ('3370', 159156), ('5296', 132136)] 
标签为第 8组，最多的5个单词为 [('3750', 242367), ('648', 202399), ('900', 92207), ('6122', 57345), ('4939', 56147)] 
标签为第 9组，最多的5个单词为 [('3750', 178783), ('648', 157291), ('900', 70680), ('7328', 46477), ('6122', 43411)] 
标签为第10组，最多的5个单词为 [('3750', 180259), ('648', 114512), ('900', 75185), ('3370', 67780), ('2465', 45163)] 
标签为第11组，最多的5个单词为 [('3750', 83834), ('648', 67353), ('900', 37240), ('4939', 18591), ('6122', 18438)] 
标签为第12组，最多的5个单词为 [('3750', 87412), ('4464', 51426), ('3370', 45815), ('648', 37041), ('2465', 36610)] 
标签为第13组，最多的5个单词为 [('3750', 33796), ('648', 26867), ('900', 11263), ('4939', 9651), ('669', 8925)] 


# 句尾分析
last_word_count=sentence_train.last_word_ana(2,3)
# output
最多的2个字符:
[('900', 85040), ('2662', 39273)]
最少的3个字符:
[('3104', 1), ('6832', 1), ('4304', 1)]
所有文档中不同的最后一个字符数： 1897

```
- 对于测试集：
```python
test_path="../data/test_a.csv"
sentence_test=SentenceAnalysis(test_path,n_classes=14,with_label=False)
```
```python
# 功能展示
# __getitem__
sentence_test[1]
# output
text    2491 4109 1757 7539 648 3695 3038 4490 23 7019...
Name: 1, dtype: object

# __len__
len(sentence_test)
# output
50000

# data
sentence_test.data
# output
略，是个DataFrame

# 文章长度分析
df_length=sentence_test.passage_length_ana()
# output
count    50000.000000
mean       909.844960
std       1032.313375
min         14.000000
25%        370.000000
50%        676.000000
75%       1133.000000
max      41861.000000
Name: text_len, dtype: float64


# 辅助的作图
sentence_test.show_hist(df_length,100,'Text char count',"Histogram of char count")
# output
略

# 新闻类别分布(没有标签，给出提示不可做分析。)
sentence_test.label_distribution()
# output
没有可用的标签！


# 字符个数分布
word_dict=sentence_test.word_distribution(5)
# output
最多的5个字符:
[('3750', 1879488), ('648', 1232522), ('900', 818765), ('3370', 511436), ('6122', 402213)]
最少的1个字符:
[('1224', 1)]
所有文档中拥有字符数： 6203

# 不同字符在句子中出现的次数  #(2,3)只是个示例
word_in_sentece_dict=sentence_test.word_in_sentece_distribution(2,3)
# output
最多的2个字符:
字符编号为 3750 在所有句子中的比例为: 98.91%
字符编号为  900 在所有句子中的比例为: 98.73%
最少的3个字符:
字符编号为 1876 在所有句子中的比例为: 0.00%
字符编号为 1224 在所有句子中的比例为: 0.00%
字符编号为 2436 在所有句子中的比例为: 0.00%


# 统计每类标签中出现次数最多的字符(没有标签，给出提示不可做分析。)
word_group_count=sentence_test.word_groupbylabel_count(5)
# output
没有可用的标签！

# 句尾分析
last_word_count=sentence_test.last_word_ana(2,3)
# output
最多的2个字符:
[('900', 21056), ('2662', 10021)]
最少的3个字符:
[('3577', 1), ('4302', 1), ('1832', 1)]
所有文档中不同的最后一个字符数： 1141
```
