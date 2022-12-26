#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2022/11/01 09:54:05
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   配置数据处理类及函数
'''

from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"


""" 词表类 """
# 定义一个词表类型
# 该类用于实现token到索引的映射
class Vocab:

    def __init__(self, tokens = None) -> None:
        # 构造函数
        # tokens：全部的token列表

        self.idx_to_token = list()
        # 将token存成列表，索引直接查找对应的token即可
        self.token_to_idx = dict()
        # 将索引到token的映射关系存成字典，键为索引，值为对应的token

        if tokens is not None:
            # 构造时输入了token的列表
            if "<unk>" not in tokens:
                # 不存在标记
                tokens = tokens + "<unk>"
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                # 当前该token对应的索引是当下列表的最后一个
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        # 构建词表
        # cls：该类本身
        # text：输入的文本
        # min_freq：列入token的最小频率
        # reserved_tokens：额外的标记token
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        # 统计各个token的频率
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        # 加入额外的token
        uniq_tokens += [token for token, freq in token_freqs.items() \
            if freq >= min_freq and token != "<unk>"]
        # 全部的token列表
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入token对应的索引，不存在则返回<unk>返回的索引
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标签对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in ids]

""" 数据加载类 """
class Load_dataset():

    def load_sentence_polarity():

        from nltk.corpus import sentence_polarity

        vocab = Vocab.build(sentence_polarity.sents())
        # 使用nltk的情感倾向数据作为示例

        train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories="pos")[:4000]]\
            +[(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[:4000]]
        # 分别取褒贬各4000句作为训练数据，将token映射为对应的索引值

        test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories="pos")[4000:]]\
            +[(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories="neg")[4000:]]
        # 其余数据作为测试数据

        return train_data, test_data, vocab

    def load_treebank():

        # 使用宾州树库词性标注数据库
        from nltk.corpus import treebank

        sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))
        # sents存储全部经过标记化的句子
        # postags存储每个标记对应的词性标注结果
        vocab= Vocab.build(sents, reserved_tokens=["<pad>"])
        # 使用pad标记来补齐序列长度
        tag_vocab = Vocab.build(postags)
        # 将词性标注标签也映射为索引值
        train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags))\
            for sentence, tags in zip(sents[:3000], postags[:3000])]
        # 取前3000句作为训练数据，将token映射为对应的索引值
        test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags))\
            for sentence, tags in zip(sents[3000:], postags[3000:])]
        # 其余的作为测试数据

        return train_data, test_data, vocab, tag_vocab
    
    def load_reuters():
        # 从nltk中导入reuters数据
        from nltk.corpus import reuters
        # 获取reutuers数据
        text = reuters.sents()
        # 将字母都转化为小写
        text = [[word.lower() for word in sentence] for sentence in text]
        # 构建词表
        vocab = Vocab.build(text, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
        # 将文本数据转换为id表示
        corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]
        return vocab, corpus
    
''' 数据集类 '''
# 声明一个DataSet类
class BowDataset(Dataset):

    def __init__(self, data) -> None:
        # data：使用load_sentence_polarity获得的数据
        self.data = data

    def __len__(self):
        # 返回样例的数目
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

'''用于词向量生成的数据集类'''
# 用于前馈神经网络词向量生成的Dataset
class NGramDataset(Dataset):
    
    def __init__(self, corpus, vocab, context_size = 2):

        self.data = []
        self.bos = vocab[BOS_TOKEN]# 句首标记
        self.eos = vocab[EOS_TOKEN]# 句尾标记

        for sentence in tqdm(corpus, desc = "Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos] # 插入句首句尾标记符
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入：长度为context_size的上下文
                context = sentence[i-context_size:i]
                # 模型输出：当前词
                target = sentence[i]
                # 每个训练样本由(context, target)组成
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# 用于RNN词向量生成的Dataset
class RnnlmDataset(Dataset):
    
    def __init__(self, vocab, corpus):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc = "Dataset Construction"):
            # 输入序列：BOS_TOKEN，w1，w2......
            input = [self.bos] + sentence
            # 输出序列：w1，w2，EOS_TOKEN
            target = sentence + [self.eos]
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 用于CBOW模型的数据集
class CbowDataset(Dataset):

    def __init__(self, vocab, corpus, context_size = 2) -> None:
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size * 2 + 1:
                # 此处的context_size是单向上下文长度，因此，如果小于上文长度，无法构建该任务
                continue
            for i in range(context_size, len(sentence) - context_size):
                context = sentence[i-context_size:i] + sentence[i+1:i+context_size]
                # 模型输入：左右各取context_size的上下文
                target = sentence[i]
                # 模型输出：中间的单词
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 定义用于Skip-gram模型的数据集
class SkipGramDataset(Dataset):

    def __init__(self, vocab, corpus, context_size=2) -> None:
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 从第二个单词开始，到倒数第二个词为之
                w = sentence[i]
                # 模型输入：当前词
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index]
                # 模型输出：上下文窗口内的共现词，如果窗口边缘超出了字符串左右，则截取到字符串尽头
                self.data.extend([(w, c) for c in context])
                # 此处使用extend，因为该列表里面是多个词对，每一个是一个输出

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SGNSDataset(Dataset):
# 用于负采样的Skip-Gram的数据集
# 我们在数据集构建时生成负样本
    def __init__(self, vocab, corpus, context_size=2, n_negatives=5, ns_dist = None) -> None:
        # n_negative指生成负样本个数
        # ns_dist指生成负样本分布，None为均匀分布
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc = "Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                w = sentence[i]
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index]
                # 模型输入为当前词和上下文的词对
                context += [self.pad] * (2 * context_size - len(context))
                # 需要对上下文进行补齐
                self.data.append((w, context))
                # 输出为0/1，标志是否为负样本

        self.n_negatives = n_negatives
        self.ns_dist = ns_dist if ns_dist == None else torch.ones(len(vocab))
        # None为均匀分布

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
