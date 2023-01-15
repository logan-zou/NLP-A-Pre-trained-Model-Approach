#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2023/01/15 10:30:31
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   用于ELMo模型的数据加载及处理
'''

from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from tqdm import tqdm

# 创建文本预定义标记
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOW_TOKEN = "<bow>"
EOW_TOKEN = "<eow>"

# 构建Vocab类
class Vocab:

    def __init__(self, tokens = None) -> None:
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens += ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx["<unk>"] 

    @classmethod
    def build(cls, text, min_freq = 1, reserved_tokens = None):
        # cls 为类本身，相当于Vocab()
        token_freqs = defaultdict(int) # 统计token的频率
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items()  
                       if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)
        
    def __len__(self):
        # 返回词表的大小
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入token对应的索引值，如果不存在返回<unk>对应的索引0
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]

# 创建词表函数
def load_corpus(path, max_tok_len=None, max_seq_len=None):
    '''
    path:本地文本数据路径
    max_tok_len:词长度上限
    max_seq_len:序列长度上限
    '''
    text = []
    charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN}
    # 字符集，首先加入预定义标记
    with open(path, "r") as f:
        # 读取文本文件
        for line in tqdm(f):
            # 文件中每一行是一段字符序列
            tokens = line.rstrip().split(" ")
            # rstrip函数用于删除字符串末尾的空白
            if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
                # 之后要加入BOS_TOKEN和EOS_TOKEN两个标记，所以要留出两个位置
                tokens = line[:max_seq_len-2]
                # 截断过长的序列
            sent = [BOS_TOKEN]
            # 当前序列
            for token in tokens:
                if max_tok_len is not None and len(token) + 2 > max_tok_len:
                    # 同理，要加入BOW_TOKEN和EOW_TOKEN
                    # 注意，因为ELMo模型使用了字符级输入，所以除构建词级语料外，还要构建字符级语料
                    token = token[:max_tok_len-2]
                sent.append(token)
                for ch in token:
                    charset.add(ch)
                    # 将字符加入字符集
            sent.append(EOS_TOKEN)
            text.append(sent)
    # 此处处理之后，text中的一个元素为一个序列即一个sent，sent中的一个元素为一个标注即一个token
    # print(text[:10])
    
    vocab_w = Vocab.build(text, min_freq=2, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    # 词级词表，需要先统计token，因此使用build方法
    vocab_c = Vocab(tokens=list(charset))
    # 字符级词表，charset是已经统计好的字符，因此无需统计，直接构建

    corpus_w = [vocab_w.convert_tokens_to_ids(sent) for sent in text]
    # 构建词级语料
    corpus_c = []
    bow = vocab_c[BOW_TOKEN]
    eow = vocab_c[EOW_TOKEN]
    for i, sent in enumerate(text):
        sent_c = []
        for token in sent:
            if token == BOS_TOKEN or token == EOS_TOKEN:
                token_c = [bow, vocab_c[token], eow]
            # 如果token不是一个词
            else:
                token_c = [bow] + vocab_c.convert_tokens_to_ids(token) + [eow]
            sent_c.append(token_c)
        # 这一块代码整体是将文本转化为索引，对于正常token，直接调用转化函数即可，对于标记类token，则直接查找对应索引
        # 因为convert函数内部会将传入参数拆开依次映射，因此vocab_c传入token，但其实映射的是token内部的字符
        corpus_c.append(sent_c)

    return corpus_w, corpus_c, vocab_w, vocab_c

# 创建用于双向语言模型的数据集
class BiLMDataset(Dataset):

    def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c) -> None:
        super(BiLMDataset, self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data = []
        for sent_w, sent_c in zip(corpus_w, corpus_c):
            self.data.append((sent_w, sent_c))
        # print(self.data[0][1])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# 针对ELMo模型的数据采样函数
def collate_fn(examples, pad_w, pad_c):
    
    seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])
    # ex为每段文本，ex[0]为以词划分的序列
    # 样本中序列的长度，使用LongTensor函数进行数据类型的转换
    inputs_w = [torch.tensor(ex[0]) for ex in examples]
    # 词级别输入
    inputs_w = pad_sequence(inputs_w, batch_first=True, padding_value=pad_w)
    # 对每个序列补齐到相同长度

    batch_size, max_seq_len = inputs_w.shape
    # 词级别的输入矩阵为批次大小*序列长度，因为之前做了补齐，所以所有长度皆为最长序列长度
    max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])
    # ex[1]为以字符划分的序列，tok为以字符表示的每个词
    # 找出最大词大小

    inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(pad_c)
    # 字符级别的输入矩阵为批次大小*序列长度*最大词大小,使用pad初始化
    # 字符比词更深一层
    for i, (sent_w, sent_c) in enumerate(examples):
        for j, tok in enumerate(sent_c):
            inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)
            # 此处使用索引起到了补齐的作用

    targets_fw = torch.LongTensor(inputs_w.shape).fill_(pad_w)
    # 前向语言模型的目标输出序列
    targets_bw = torch.LongTensor(inputs_w.shape).fill_(pad_w)
    # 后向语言模型的目标输出序列
    for i, (sent_w, sent_c) in enumerate(examples):
        targets_fw[i][:len(sent_w)-1] = torch.LongTensor(sent_w[1:])
        # 前向语言模型的目标输出序列为输入序列左移一位
        targets_bw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w)-1])
    # 对于前向语言模型，输入为<bos>w1w2w3...<eos>，输出为w1w2w3...<eos><pad>
    # 计算时输入<bos>输出w1，输入w1和历史状态（<bos>）输出w2，以此类推
    # 对于后向语言模型，输入为<bos>w1w2w3...<eos>，输出为<pad><bos>w1w2...wn

    return inputs_w, inputs_c, seq_lens, targets_fw, targets_bw