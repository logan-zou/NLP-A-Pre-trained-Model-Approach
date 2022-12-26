#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vec_generate.py
@Time    :   2022/12/26 22:01:33
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   用于生成词向量的文件,此处以RNN为例
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from data_process import Load_dataset
from data_process import RnnlmDataset
from nlp_collate import Vec
from nlp_module import RNNLM
from network_tools import save_pretrained


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

# 训练
# 超参数设置
hidden_dim = 256 # 隐藏层维度
context_size = 3 # 上下文长度
num_epoch = 10 # 迭代次数
embedding_dim = 128
batch_size = 16

# 加载数据
vocab, corpus = Load_dataset.load_reuters()
dataset = RnnlmDataset(vocab, corpus)
data_loader = DataLoader(dataset, batch_size, collate_fn=lambda x : Vec.RNN_collate_fn(x, vocab[PAD_TOKEN]))

# 设置ignore_index参数，以忽略PAD_TOKEN处的损失
nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
model = RNNLM(len(vocab), embedding_dim, hidden_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc = f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss:{total_loss:.2f}")
    
save_pretrained(vocab, model.embedding.weight.data, "RNNLM.vec")