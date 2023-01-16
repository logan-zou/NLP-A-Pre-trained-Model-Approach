#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   reading_comprehension.py
@Time    :   2023/01/16 20:16:36
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   该文件为基于transformers实现对基准数据集SQUAD的抽取式阅读理解
'''

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

# 准备训练数据
'''这一部分没有完全理解，该部分使用了较多专用API，由于本人并不专攻阅读理解任务，暂不深入思考'''
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples['question'], # 问题文本
        examples['context'], # 篇章文本
        truncation = 'only_second', # 截断只发生在第二部分，即只截断篇章文本
        max_length = 384, # 最大长度
        stride = 128, # 篇章切片步长
        return_overflowing_tokens = True, # 返回超出最大长度的标记，将篇章切成多片
        return_offsets_mapping = True, # 返回偏置信息，用于对齐答案位置
        padding = 'max_length' # 按最大长度补齐
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # 建立到example的映射关系
    offset_mapping = tokenized_examples.pop("offset_mapping")
    # 建立token到原文的字符级映射关系，用于确定答案的开始位置和结束位置

    tokenized_examples["start_position"] = []
    tokenized_examples["end_position"] = []
    # 开始位置和结束位置

    for i, offsets in enumerate(offset_mapping):
        # 遍历输入序列
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # 获取输入序列的input_ids以及[cls]标记的位置

        sequence_ids = tokenized_examples.sequence_ids(i)
        # 获取哪些部分是问题，哪些部分是篇章

        sample_index = sample_mapping[i]
        # 第i个序列
        answers = examples["answers"][sample_index]
        # 答案
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        # 答案的开始结束位置

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        # 找到值为1的位置
        token_end_index = 0
        while sequence_ids[token_end_index] != 1:
            token_end_index += 1
        # 同上，应该是值为1标志着答案开始和结束

        if not(offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # 答案是否超出当前切片的范围
            tokenized_examples["start_position"].append(cls_index)
            tokenized_examples["end_position"].append(cls_index)
            # 如果超出，开始和结束的位置均设置为cls的位置
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index +=1
            tokenized_examples["start_position"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_position"].append(token_end_index + 1)
            # 将开始和结束位置移至篇章中答案的两端

    return tokenized_examples

# 加载训练数据
dataset = load_dataset('squad')
# 此函数直接从网络下载基准数据集
# 加载分词器、预训练模型和评价方法
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
# 此处的参数可以选择不同的分词器，可以在hagging face 官网查阅
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict = True)
# 模型选择同理，还可以选择一些学者上传的、特别的预训练模型
metric = load_metric('/home/zouyuheng/tool/huggingface-datasets/squad.py')
# 基准数据集是有特定的评价指标的
# 注意，此处mertic的加载无法直接联网获取，需要将对应文件下载到本地，输入本地文件路径获取

tokenize_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)

# 设置训练超参
args = TrainingArguments(
    "ft-squad", # 输出路径
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
    data_collator = default_data_collator
)

trainer.train()