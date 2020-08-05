# nlp_learning
 在datawhale学习nlp文本分类的学习经过中书写的代码，希望自己以后可以循环利用
 
 学习笔记：[博客](https://www.yuque.com/zzhijiki/ssgfub)
 
 项目任务：针对天池大赛的匿名字符集，构建不同的文本分类模型，来进行文本分类任务的实践。

## Task 02：

建立了一个文章分析类，是一个数据分析和可视化的项目。

详细的功能有：

- 文章长度分析
- 类别分布分析
- 字符个数分布分析
- 不同字符在句子中出现的次数
- 统计每类标签中出现次数最多的字符
- 句尾分析



## Task 04：

针对数据集建立了一个FastText分类器，这是一个三层的神经网络，但是不包括论文中的ngram和分层softmax。

具体的过程为：

1. 我们针对这次的匿名数据集，使用`torchtext`进行数据的预处理
2. 搭建了一个最简单的FastText的模型，但它不包含ngram和分层softmax。
3. 构造了训练，验证，预测的整个迭代过程

总体来说，是一个比较完整的简单的项目，提交之后

线下得分0.902

线上得分0.902。



## Task 05:

### Word2vec

根据匿名数据集进行了 word2vec 的预训练，将预训练的词向量加入之后建立的分类器，进行预测。

### TextCNN

建立了一个TextCNN分类器，提交之后

线下分数0.938，

线上分数0.937。


### TextRNN

建立了一个TextRNN分类器，由RNN可以输入变长序列的性质，我们建立了等长（被padding）和变长两种模型。

有趣的是，变长需要用RMSProp才能train 起来。

TextRNN等长模型 线下分数：0.932

TextRNN变长模型 线下分数：0.928

线上没有提交。

### HAN

HAN 的文本层级结构为 词→句→文，根据这个特点，重新构建了Dataset类。

建立了一个基于Attention 的 HAN 分类器。

线下得分：0.9438

线上得分：0.9426，冲进了前15名，当然，提分技巧很多，可以继续尝试。

## Task 06:

建立了基于bert模型的分类器

线下得分：0.9451

线上得分：0.9449，重返前20名
