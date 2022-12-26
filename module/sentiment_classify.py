#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sentiment_classify.py
@Time    :   2022/11/01 10:41:52
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   用于情感分类的训练文件，此处以Transformer为例
'''

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim


from data_process import Load_dataset
from data_process import BowDataset
from nlp_collate import Sentiment_classify
from nlp_module import Transformer

# 训练
# 超参数设置
embedding_dim = 128
batch_size = 16
num_epoch = 10
num_class = 2

train_data, test_data, vocab = Load_dataset.load_sentence_polarity()
# 加载数据
train_dataset = BowDataset(train_data)
test_dataset = BowDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=Sentiment_classify.lstm_collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=Sentiment_classify.lstm_collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(len(vocab), embedding_dim, num_class)
model.to(device)
# 加载模型

nll_loss = nn.NLLLoss()
# 负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        # print(inputs.size())
        # print("inputs", inputs.is_cuda)
        # print("lengths", lengths.is_cuda)
        log_probs = model(inputs, lengths)
        # print("log_probs", log_probs.size())
        # print("targets, ", targets.size())
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss:{total_loss:.2f}")

    # 测试
    acc = 0
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths)
            acc += (output.argmax(dim=1) == targets).sum().item()
    print(f"ACC:{acc / len(test_data_loader):.2f}")