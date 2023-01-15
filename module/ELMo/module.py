#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   module.py
@Time    :   2023/01/15 10:33:13
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   用于ELMo模型的模型定义
'''

from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import os
import torch
'''Highway神经网络'''
class Highway(nn.Module):
    # 基于字符的输入表示层，即Highway网络
    def __init__(self, input_dim, num_layers, activation = F.relu) -> None:
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        # 使用ModuleList构建多个线性层，每层的输入为input_dim，输出为两倍input_dim，其中一半输入下一层，一半用于计算门控
        self.activation = activation
        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)
        # 后半部分是计算门控向量的参数,即公式中的W^g和b^g

    def forward(self, inputs):
        curr_inputs = inputs
        # 整体输入
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)
            # 经过线性层计算
            hidden = self.activation(projected_inputs[:, 0:self.input_dim])
            # 前半部分通过激活作为当前隐藏层输出
            gate = torch.sigmoid(projected_inputs[:, self.input_dim:])
            # 后半部分计算门控向量
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        return curr_inputs

'''字符卷积层'''
class ConvTokenEmbedder(nn.Module):
    # 基于字符卷积的词表示层
    def __init__(self, vocab_c, char_embedding_dim, char_conv_filters, num_highways, output_dim, pad = "<pad>") -> None:
        '''
        Args:
           vocab_c: 字符级词表
           char_embedding_dim: 字符向量维度
           char_conv_filters: 卷积核大小，双层列表
           num_highways: highway网络层数 
           output_dim: 输出维度
        '''
        super(ConvTokenEmbedder, self).__init__()
        self.vocab_c = vocab_c
        self.char_embeddings = nn.Embedding(len(vocab_c), char_embedding_dim, padding_idx=vocab_c[pad])
        # 词向量层，注意，此处的len(vocab_c)并不是输入维度，而是词表大小
        # self.char_embeddings.data.uniform(-0.25, 0.25)
        # uniform随机取值，对参数进行初始化

        self.convolutions = nn.ModuleList()
        # 卷积层
        for kernel_size, out_channels in char_conv_filters:
            conv = nn.Conv1d(in_channels=char_embedding_dim, out_channels=out_channels, kernel_size=kernel_size, bias=True)
            self.convolutions.append(conv)
        # 创建多个一维卷积层

        self.num_filters = sum(f[1] for f in char_conv_filters)
        # 输入highway网络时是将不同卷积核的输出拼接在了一起，所以highway网络的输入维度是所有卷积层的输出维度之和
        self.num_highways = num_highways
        self.highways = Highway(self.num_filters, self.num_highways, activation = F.relu)
        self.projection = nn.Linear(self.num_filters, output_dim, bias = True)
        # 线性规整层

    def forward(self, inputs):
        batch_size, seq_len, token_len = inputs.shape
        # 批次大小，序列长度，标记长度
        # print("inputs.shape = ", inputs.shape)
        inputs = inputs.view(batch_size * seq_len, -1)
        # 将输入展开为二维，以token为单位，所以需要将批次和序列长度整合
        # print("after view, inputs.shape = ", inputs.shape)
        # print("len(vocab_c) = ", len(self.vocab_c))
        char_embeds = self.char_embeddings(inputs)
        # print("char_embeds.shape = ", char_embeds.shape)
        char_embeds = char_embeds.transpose(1, 2)
        # 做转置原因：卷积的输入定义不同，为批次*输入通道数*长度，表示层输出为批次*长度*输入通道数
        # 注意，由于将token拆成了字符，此处的批次其实为批次*token数

        conv_hiddens = []
        for i in range(len(self.convolutions)):
            # 逐个卷积操作
            conv_hidden = self.convolutions[i](char_embeds)
            conv_hidden, _ = torch.max(conv_hidden, dim = -1)
            # 最大池化
            conv_hidden = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)

        token_embeds = torch.cat(conv_hiddens, dim=-1)
        # 将不同卷积层的输出拼接
        token_embeds = self.highways(token_embeds)
        token_embeds = self.projection(token_embeds)
        token_embeds = token_embeds.view(batch_size, seq_len, -1)
        # 将输出的形状还原
        return token_embeds

'''BiLSTM-Encoder'''
class ELMoLSTMEncoder(nn.Module):
    # 双向LSTM编码器
    def __init__(self, input_dim, hidden_dim, num_layers) -> None:
        
        super(ELMoLSTMEncoder, self).__init__()
        self.projection_dim = input_dim
        # 用于投影层，保证各层具有相同的维度
        self.num_layers = num_layers

        self.forward_layers = nn.ModuleList()
        # 前向LSTM
        self.forward_projections = nn.ModuleList()
        # 投影层：hidden_dim -> projection_dim
        self.backward_layers= nn.ModuleList()
        # 后向LSTM
        self.backward_projections = nn.ModuleList()
        # 后向投影层同前向

        lstm_input_dim = input_dim
        for _ in range(num_layers):
            forward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)
            # 单层前向LSTM以及投影层
            backward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)
            # 单层后向LSTM以及投影层
            lstm_input_dim = self.projection_dim
            self.forward_layers.append(forward_layer)
            self.forward_projections.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projections.append(backward_projection)

    def forward(self, inputs, lengths):
        batch_size, seq_len, input_dim = inputs.shape
        
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        # print("最初的rev_idx")
        # print(rev_idx)
        for i in range(lengths.shape[0]):
            rev_idx[i, :lengths[i]] = torch.arange(lengths[i]-1, -1, -1)
            # print("经过第{}次处理之后的rev_idx")
            # print(rev_idx)
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        # print("处理之后的rev_idx")
        # print(rev_idx)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1, rev_idx)
        '''此处不是特别清晰'''

        forward_inputs, backward_inputs = inputs, rev_inputs
        # 前向和后向的输入
        stacked_forward_states, stacked_backward_states = [], []
        # 前向和后向的隐含层状态

        for layer_index in range(self.num_layers):
            
            packed_forward_inputs = pack_padded_sequence(forward_inputs, lengths, batch_first=True, enforce_sorted=False)
            packed_backward_inputs = pack_padded_sequence(backward_inputs, lengths, batch_first=True, enforce_sorted=False)
            # 对前后向输入进行打包对齐
            
            forward_layer = self.forward_layers[layer_index]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            # 对输出解包
            forward = self.forward_projections[layer_index](forward)
            # 规整宽度
            stacked_forward_states.append(forward)
            # 计算前向LSTM

            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
            backward = self.backward_projections[layer_index](backward)
            stacked_backward_states.append(backward.gather(1, rev_idx))
            # 将输出还原顺序
            # 计算后向LSTM

        return stacked_forward_states, stacked_backward_states 

'''BiLSTM'''
class BiLM(nn.Module):
    def __init__(self, configs, vocab_w, vocab_c) -> None:
        super(BiLM, self).__init__()
        self.dropout_prob = configs["dropout"]
        self.num_classes = len(vocab_w)
        # 输出层的维度为词表大小，即对词表中的每一个词有一个预测概率

        self.token_embedder = ConvTokenEmbedder(vocab_c, configs['char_embedding_dim'], configs['char_conv_filters'], configs['num_highways'], configs['projection_dim'])
        # 词表示编码器
        self.encoder = ELMoLSTMEncoder(configs['projection_dim'], configs['hidden_dim'], configs['num_layers'])
        # ELMo编码器
        self.classifier = nn.Linear(configs['projection_dim'], self.num_classes)
        # 分类器

    def forward(self, inputs, lengths):
        token_embeds = self.token_embedder(inputs)
        token_embeds = F.dropout(token_embeds, self.dropout_prob)
        # 采样
        forward, backward = self.encoder(token_embeds, lengths.to('cpu'))
        return self.classifier(forward[-1]), self.classifier(backward[-1])
        # 使用最后的隐藏状态作为分类器的输入
        # 此处注意，ELMo模型的原思想应该是对各个隐藏状态做线性组合，此处做了简化

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.token_embedder.state_dict(), os.path.join("token_embedder.pth"))
        # 保存词表示编码器的参数
        torch.save(self.encoder.state_dict(), os.path.join("encoder.pth"))

