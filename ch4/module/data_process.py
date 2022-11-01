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