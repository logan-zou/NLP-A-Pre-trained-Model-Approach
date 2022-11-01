#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nlp_module.py
@Time    :   2022/11/01 10:04:05
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   用于nlp的各种神经网络模型集合
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import network_tools

''' 用于结构化数据的MLP类 '''
class MLP(nn.Module):
    # 基类为nn.Module
    def __init__(self, input_dim, hidden_dim, num_class):
        # 构造函数
        # input_dim:输入数据维度
        # hidden_dim:隐藏层维度
        # num_class:多分类个数
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # 隐含层，线性变换
        self.activate = F.relu
        # 使用relu函数作为激活函数：小于0的值输出为0
        self.linear2 = nn.Linear(hidden_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs):
        # 前向计算函数
        # inputs:输入
        # print(f"输入为：{inputs}")
        hidden = self.linear1(inputs)
        # print(f"经过隐含层变换为：{hidden}")
        activation = self.activate(hidden)
        # print(f"经过激活后为：{activation}")
        outputs = self.linear2(activation)
        # print(f"输出层输出为：{outputs}")
        probs = F.softmax(outputs, dim = 1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return probs

'''用于文本处理的MLP类'''
# 创建一个MLP类
class MLP(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim:隐藏层维度
        # num_class:多分类个数
        super(MLP, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        # 词向量层
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        # 隐含层，线性变换
        self.activate = F.relu
        # 使用relu函数作为激活函数：小于0的值输出为0
        self.linear2 = nn.Linear(hidden_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, offsets):
        # 前向计算函数
        # inputs:输入
        # print(f"输入为：{inputs.size()}")
        embeds = self.embedding(inputs, offsets)
        # 对词向量层取袋模型
        hidden = self.linear1(embeds)
        # print(f"经过隐含层变换为：{hidden}")
        activation = self.activate(hidden)
        # print(f"经过激活后为：{activation}")
        outputs = self.linear2(activation)
        # print(f"输出层输出为：{outputs}")
        probs = F.log_softmax(outputs, dim = 1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return probs


'''用于结构化数据的CNN类'''
class CNN(nn.Module):
    # 基类为nn.Module
    def __init__(self, input_dim, output_dim, num_class, kernel_size):
        # 构造函数
        # input_dim:输入数据维度
        # output_dim:卷积输出维度
        # num_class:多分类个数
        # kernel_size：卷积核宽度
        super(CNN, self).__init__()

        self.conv = Conv1d(input_dim, output_dim, kernel_size)
        # 卷积层
        self.pool = F.max_pool1d
        # 池化层，使用最大池化
        self.linear = nn.Linear(output_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs):
        # 前向计算函数
        # inputs:输入
        # print(f"输入size为：{inputs.size()}")
        conv = self.conv(inputs)
        # print(f"经过卷积层变换size为：{conv.size()}")
        pool = self.pool(conv, kernel_size = conv.shape[2])
        # print(f"经过池化后size为：{pool.size()}")
        pool_squeeze = pool.squeeze(dim=2)
        outputs = self.linear(pool_squeeze)
        # print(f"输出层输出size为：{outputs.size()}")
        return outputs

''' 用于情感分类的CNN类 '''
class CNN(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # filter_size：卷积核大小
        # num_filter: 卷积核个数
        # num_class:多分类个数
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        # 卷积层，使用1作为padding
        self.activate = F.relu
        # 使用relu函数作为激活函数：小于0的值输出为0
        self.linear = nn.Linear(num_filter, num_class)
        # 输出层，线性变换

    def forward(self, inputs):
        # 前向计算函数
        # inputs:输入
        # print(f"输入为：{inputs.size()}")
        embeds = self.embedding(inputs).permute(0, 2, 1)
        # 注意这儿是词向量层，不是词袋词向量层
        # 卷积层的输入两个维度与词向量层输出相反，需要使用permute转换一下
        # print(f"词向量层输出为：{embeds.size()}")
        convolution = self.conv1d(embeds)
        # print(f"经过卷积层变换为：{convolution.size()}")
        activation = self.activate(convolution)
        # print(f"经过激活后为：{activation.size()}")
        pooling = F.max_pool1d(activation, kernel_size=activation.shape[-1])
        # print(f"池化后为：{pooling.size()}")
        # print(f"池化后结果为：{pooling}")
        outputs = self.linear(pooling.squeeze(dim=2))
        # 池化后的输出是二维的，需要使用squeeze降维到一维
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = 1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return log_probs

''' 用于情感分类的LSTM模型 '''
# 创建一个LSTM类
class LSTM(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim：隐藏层维度
        # num_class:多分类个数
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        # lstm层
        self.output = nn.Linear(hidden_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, lengths):
        # 前向计算函数
        # inputs:输入
        # lengths:打包的序列长度
        # print(f"输入为：{inputs.size()}")
        embeds = self.embedding(inputs)
        # 注意这儿是词向量层，不是词袋词向量层
        # print(f"词向量层输出为：{embeds.size()}")
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        # LSTM需要定长序列，使用该函数将变长序列打包
        # print(f"经过打包为：{x_pack.size()}")
        hidden, (hn, cn) = self.lstm(x_pack)
        # print(f"经过lstm计算后为：{hn.size()}")
        outputs = self.output(hn[-1])
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = -1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return log_probs

''' 用于词性标注的LSTM模型 '''
# 创建一个LSTM类
class LSTM_POS(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim：隐藏层维度
        # num_class:多分类个数
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        # lstm层
        self.output = nn.Linear(hidden_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, lengths):
        # 前向计算函数
        # inputs:输入
        # lengths:打包的序列长度
        # print(f"输入为：{inputs.size()}")
        embeds = self.embedding(inputs)
        # 注意这儿是词向量层，不是词袋词向量层
        # print(f"词向量层输出为：{embeds.size()}")
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        # LSTM需要定长序列，使用该函数将变长序列打包
        # print(f"经过打包为：{x_pack.size()}")
        hidden, (hn, cn) = self.lstm(x_pack)
        # print(f"经过lstm计算后为：{hn.size()}")
        hidden, _ = pad_packed_sequence(hidden, batch_first = True)
        # 词性标注需要再进行解包，还原成经过补齐的序列
        # print(f"解包之后输出为：{hidden.size()}")
        outputs = self.output(hidden)
        # 在词性标注中需要使用全部的隐藏层状态
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = -1)
        # print(f"输出概率值为：{log_probs}")
        # 归一化为概率值
        return log_probs


''' 用于情感分类的Transformer模型 '''
# 此处书中代码有误，不需要hidden_dim，注意力层输入维度应该直接是词向量维度
class Transformer(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, num_class,
    dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str = "relu"):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim：隐藏层维度
        # num_class:多分类个数
        # dim_feedforward：前馈网络模型的维度
        # num_head:头数
        # num_layers：注意力层数
        # dropout:辍学比例
        # max_len:序列最大长度
        # activation:激活函数
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.position_embedding = network_tools.PositionalEncoding(embedding_dim, dropout, max_len)
        # 位置编码层
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation)
        # 一个encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 注意力编码层
        self.output = nn.Linear(embedding_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, lengths):
        # 前向计算函数
        # inputs:输入
        # lengths:打包的序列长度
        # print(f"输入为：{inputs.size()}")
        inputs = torch.transpose(inputs, 0, 1)
        # 首先需要将输入第一维与第二维互换，适应transformer的输入
        embeds = self.embedding(inputs)
        # 注意这儿是词向量层，不是词袋词向量层
        # print(f"词向量层输出为：{embeds.size()}")
        embeds = self.position_embedding(embeds)
        # 加入位置编码
        # print(f"位置编码层输出为：{embeds.size()}")
        attention_mask = network_tools.length_to_mask(lengths) == False
        # 生成mask掩码
        # print(f"生成mask为：{attention_mask.size()}")
        hidden_states = self.transformer(embeds, src_key_padding_mask = attention_mask)
        # 用来遮蔽<PAD>以避免pad token的embedding输入
        # print(f"经过transformer计算后为：{hidden_states.size()}")
        hidden_states = hidden_states[0, :, :]
        # 取第一个标记的输出结果作为分类层的输入
        outputs = self.output(hidden_states)
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = -1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return log_probs

''' 用于词性标注的Transformer模型 '''
# 创建一个Transformer类
# 此处书中代码有误，不需要hidden_dim，注意力层输入维度应该直接是词向量维度
class Transformer_POS(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, num_class,
    dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str = "relu"):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim：隐藏层维度
        # num_class:多分类个数
        # dim_feedforward：前馈网络模型的维度
        # num_head:头数
        # num_layers：注意力层数
        # dropout:辍学比例
        # max_len:序列最大长度
        # activation:激活函数
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.position_embedding = network_tools.PositionalEncoding(embedding_dim, dropout, max_len)
        # 位置编码层
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation)
        # 一个encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 注意力编码层
        self.output = nn.Linear(embedding_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, lengths):
        # 前向计算函数
        # inputs:输入
        # lengths:打包的序列长度
        # print(f"输入为：{inputs.size()}")
        inputs = torch.transpose(inputs, 0, 1)
        # 首先需要将输入第一维与第二维互换，适应transformer的输入
        embeds = self.embedding(inputs)
        # 注意这儿是词向量层，不是词袋词向量层
        # print(f"词向量层输出为：{embeds.size()}")
        embeds = self.position_embedding(embeds)
        # 加入位置编码
        # print(f"位置编码层输出为：{embeds.size()}")
        attention_mask = network_tools.length_to_mask(lengths) == False
        # 生成mask掩码
        # print(f"生成mask为：{attention_mask.size()}")
        hidden_states = self.transformer(embeds, src_key_padding_mask = attention_mask).transpose(0, 1)
        # 用来遮蔽<PAD>以避免pad token的embedding输入
        # print(f"经过transformer计算后为：{hidden_states.size()}")
        # hidden_states = hidden_states[0, :, :]
        # 取序列中每个输入的隐藏层作为分类层的输入
        outputs = self.output(hidden_states)
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = -1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return log_probs
