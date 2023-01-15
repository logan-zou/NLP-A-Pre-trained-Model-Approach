#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pre_training.py
@Time    :   2023/01/15 10:36:16
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   用于ELMo模型的预训练运行
'''

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import json
from torch import nn
import torch

from data_process import load_corpus
from data_process import BiLMDataset
from data_process import collate_fn
from module import BiLM

PAD_TOKEN = "<pad>"

config = json.load(open('config.json'))

corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(config['train_file'])
train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
train_loader = DataLoader(train_data, config['batch_size'], collate_fn = lambda x : collate_fn(x, vocab_w[PAD_TOKEN], vocab_c[PAD_TOKEN]))

criterion = nn.CrossEntropyLoss(ignore_index = vocab_w[PAD_TOKEN], reduction='sum')
# 使用交叉熵损失函数

model = BiLM(config, vocab_w, vocab_c)
# 构建模型
device = torch.device("cuda")
model.to(device)
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr = config['learning_rate'])

model.train()
for epoch in range(config['num_epoch']):
    total_loss = 0
    total_tags = 0
    # 有效预测位置的数量
    for batch in tqdm(train_loader, desc = f"Training Epoch {epoch}"):
        batch = [x.to(device) for x in batch]
        inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = batch
        optimizer.zero_grad()
        outputs_fw, outputs_bw = model(inputs_c, seq_lens)
        # 模型计算输出
        loss_fw = criterion(outputs_fw.view(-1, outputs_fw.shape[-1]), targets_fw.view(-1))
        # 计算前向模型损失
        loss_bw = criterion(outputs_bw.view(-1, outputs_bw.shape[-1]), targets_bw.view(-1))
        # 计算后向模型损失
        loss = (loss_fw + loss_bw) / 2.0
        loss.backward()
        # 反向传播
        nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        # 梯度裁剪，解决梯度爆炸问题
        optimizer.step()

    total_loss += loss_fw.item()
    total_tags += seq_lens.sum().item()
    train_ppl = np.exp(total_loss / total_tags)
    # 以前向模型的困惑度作为性能指标
    print(f"Train PPL: {train_ppl:.2f}")

model.save_pretrained(config['model_path'])
# 保存模型参数
json.dump(config, open(os.path.join(config['model_path'], 'config.json'), "w"))
# 保存超参数