#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   network_tools.py
@Time    :   2022/11/01 10:13:26
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   实现各种工具函数和类
'''

import torch.nn as nn
import torch
import math

''' 工具层的实现 '''
# 首先实现一个位置编码层
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512) -> None:
        # d_model：模型计算公式中的参数
        # dropout：辍学率
        # max_len: 事先准备好的序列长度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # 生成全零的矩阵,5000*512的矩阵，5000个位置，每个位置用一个512维度向量来表示位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 生成位置序列，unsqueeze用于升一个维度，(5000,) -> (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 公式中的w_k
        pe[:, 0::2] = torch.sin(position * div_term)
        # 偶数位置编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 奇数位置编码
        pe = pe.unsqueeze(0).transpose(0, 1)    
        # 升维，为batch_size留出位置
        self.register_buffer('pe', pe)
        # 在内存中定一个常量，即将位置编码存进去

    def forward(self, x):
        # 前向计算
        # x:输入词向量序列
        # print(x.size())
        # print(self.pe.size())
        try:
            x = x + self.pe[:x.size(0), :]
        except:
            # print(x.size())
            # print(self.pe.size())
            pass
            # 针对有一个输入会出现问题，将其跳过，不进行位置编码
        # 输入词向量与位置编码相加
        return x

''' 工具函数的实现 '''
# 定义一个根据序列长度生成Mask矩阵的函数
def length_to_mask(lengths):
    # lengths:给定序列长度
    max_len = torch.max(lengths)
    # print("maxlen", max_len.is_cuda)
    # print(torch.arange(max_len).is_cuda)
    mask = torch.arange(max_len).to("cuda").expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask