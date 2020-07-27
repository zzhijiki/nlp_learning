import gensim
import numpy as np


class ModelEmbedding:
    def __init__(self, w2v_path):
        # 加载词向量矩阵
        self.word2vector = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        # 输入的总词数
        self.vocabs = self.word2vector.vocab.keys()
        special_token_list = ["<PADDING>", "<UNKNOWN>"]
        self.input_word_list = special_token_list + self.word2vector.index2word
        self.embedding = np.concatenate((np.random.rand(2, 300), self.word2vector.vectors))
        self.dict_length = len(self.input_word_list)
        self.get_word2id()
        print("ModelEmbedding End!")

    def get_word2id(self):
        # word2id词典
        self.word2id = dict(zip(self.input_word_list, list(range(len(self.input_word_list)))))
