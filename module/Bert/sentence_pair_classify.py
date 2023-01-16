#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sentence_pair_classify.py
@Time    :   2023/01/16 18:37:17
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   该文件为基于transformers实现对基准数据集RTE的句对分类
'''

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

# 加载训练数据
dataset = load_dataset('glue', 'rte')
# 此函数直接从网络下载基准数据集
# 加载分词器、预训练模型和评价方法
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
# 此处的参数可以选择不同的分词器，可以在hagging face 官网查阅
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict = True)
# 模型选择同理，还可以选择一些学者上传的、特别的预训练模型
metric = load_metric('/home/zouyuheng/tool/huggingface-datasets/glue.py', 'rte')
# 基准数据集是有特定的评价指标的
# 注意，此处mertic的加载无法直接联网获取，需要将对应文件下载到本地，输入本地文件路径获取

# 对训练集分词
def tokenize(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation = True, padding = 'max_length')
    # 注意，由于dataset版本更新，此处键名应改为sentence1和sentence2
dataset = dataset.map(tokenize, batched = True)
# 通过tokenize对数据集进行批量处理
encoded_dataset = dataset.map(lambda example: {'label': example['label']}, batched=True)

# 数据格式转换
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
encoded_dataset.set_format(type = 'torch',  columns = columns)

# 定义评价指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions = np.argmax(predictions, axis = 1), references = labels)

# 设置训练超参
args = TrainingArguments(
    "ft-rte", # 输出路径
    evaluation_strategy = "epoch",# 每轮结束后进行评价
    learning_rate = 2e-5,
    per_device_train_batch_size = 32,# 训练批次大小
    per_device_eval_batch_size = 32,# 验证批次大小
    num_train_epochs = 2 # 训练轮次
)

# 训练
trainer = Trainer(
    model, 
    args, 
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["validation"],
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()
trainer.evaluate()