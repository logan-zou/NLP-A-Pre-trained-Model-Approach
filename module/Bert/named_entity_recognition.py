#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   named_entity_recognition.py
@Time    :   2023/01/16 21:10:54
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   201983010@uibe.edu.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   该文件为基于transformers实现对基准数据集CoNLL-2003的命名实体识别
'''

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

def tokenize_and_align_labels(examples):
    # 将训练集转化为特征形式，即分词以及对齐标签
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words = True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        # 此处应该是标签序列
        word_ids = tokenized_inputs.word_ids(batch_index = i)
        # 找到对应词的id，由于NER任务中，BERT会将token拆分，同一token拆分的子词共享同一标签，因此该标签对应了一个子词序列
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 遍历每一个子词
            if word_idx is None:
                 label_ids.append(-100)
            # 如果词为空，说明是特殊符号，将其标签设置为-100，之后在计算损失函数中将忽略
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
            # 将标签设置到每个词的第一个token上，此处就是实现子词共享标签的操作
            '''没有特别理解为什么要判断一个elif'''
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    # 设置为标签
    return tokenized_inputs

def compute_metrics(p):
    # 定义评价指标
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 2)
    # 取概率最大的作为预测结果
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
        # 取预测结果，去掉我们标注为-100的特殊符号
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
        # 取预测结果，去掉我们标注为-100的特殊符号
    ]

    results = metric.compute(predictions = true_predictions, references=true_labels)
    return {
        "precision" : results["overall_precision"],# 精确率
        "recall" : results["overall_recall"], # 召回率
        "f1" : results["overall_f1"], # f1值
        "accuracy" : results["overall_accuracy"] # 准确率
    }

dataset = load_dataset('conll2003')
# 此处如果发现数据下载不成功，可以待会儿再试
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

tokenize_datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)
'''此处为什么要将从缓存中加载设置为False'''

label_list = dataset["train"].features["ner_tags"].feature.names
# 获取所有的标签列表
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels = len(label_list))
# 加载预训练模型

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("/home/zouyuheng/tool/huggingface-datasets/seqeval.py")
# 注意，此处不光需要下载文件到本地，而且需要pip安装seqeval库

# 设置训练超参
args = TrainingArguments(
    "ft-conll2003", # 输出路径
    evaluation_strategy = "epoch",# 每轮结束后进行评价
    learning_rate = 2e-5,
    per_device_train_batch_size = 32,# 训练批次大小
    per_device_eval_batch_size = 32,# 验证批次大小
    num_train_epochs = 3 # 训练轮次
)

# 训练
trainer = Trainer(
    model, 
    args, 
    train_dataset = tokenize_datasets["train"],
    eval_dataset = tokenize_datasets["validation"],
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
    data_collator = data_collator
)

trainer.train()
trainer.evaluate()