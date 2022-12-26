import torch
from torch.nn.utils.rnn import pad_sequence
# from network_tools import Vocab

''' 用于情感分类的样本输入输出构建 '''
class Sentiment_classify():

    ''' MLP '''
    def mlp_collate_fn(examples):
        # 从独立样本集合中构建各批次的输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        # 目标targets为该批次所有样例输出结果构成的张量
        offsets = [0] + [i.shape[0] for i in inputs]
        # 一个批次中每个样例的序列长度
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        # 根据序列长度计算每个序列起始位置的偏移量
        inputs = torch.cat(inputs)
        # 将inputs列表中的张量拼接成一个大的张量
        return inputs, offsets, targets

    ''' CNN'''
    def cnn_collate_fn(examples):
        # 从独立样本集合中构建各批次的输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        # 目标targets为该批次所有样例输出结果构成的张量
        inputs = pad_sequence(inputs, batch_first=True)
        # 将用pad_sequence对批次类的样本进行补齐
        return inputs, targets

    ''' LSTM;Transformer也使用该函数 '''
    def lstm_collate_fn(examples):
        # 从独立样本集合中构建各批次的输入输出
        lengths = torch.tensor([len(ex[0]) for ex in examples])
        # 获取每个序列的长度
        inputs = [torch.tensor(ex[0]) for ex in examples]
        # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        # 目标targets为该批次所有样例输出结果构成的张量
        inputs = pad_sequence(inputs, batch_first=True)
        # 将用pad_sequence对批次类的样本进行补齐
        return inputs, lengths, targets

''' 用于词性标注的样本构建 '''
class Pos_tagging():

    '''LSTM, Transformer也使用该函数'''
    # 此处存在vocab的导入问题，可以将代码拆解使用，但封装存在问题   
    def collate_fn(examples):
        # 从独立样本集合中构建各批次的输入输出
        lengths = torch.tensor([len(ex[0]) for ex in examples])
        # 获取每个序列的长度
        inputs = [torch.tensor(ex[0]) for ex in examples]
        # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
        targets = [torch.tensor(ex[1]) for ex in examples]
        # 目标targets为该批次所有样例输出结果构成的张量，同文本分类任务不同
        inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
        targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])
        # 将用pad_sequence对批次类的样本进行补，标签也需要补齐
        return inputs, lengths, targets, inputs != vocab["<pad>"]

'''用于词向量生成的样本构建'''
class Vec():

    '''前馈神经网络使用,CBOW、Skip-gram也使用该函数'''
    def FNN_collate_fn(examples):
        # 从独立样本集合中构建批次的输入输出，并转换为PyTorch张量
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)

    '''RNN网络使用'''
    def RNN_collate_fn(examples, pad):
        # 从独立样本集合中构建批次的输入输出，并转换为PyTorch张量
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]
        # 注意此处先生成了列表，而不是如同前文一样生成tensor，因为tensor需要补齐
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad)
        targets = pad_sequence(targets, batch_first=True, padding_value=pad)
        return (inputs, targets)

    '''基于负采样的Skip-gram模型使用'''
    def SGNS_collate_fn(examples, ns_dist, n_negatives):
        # 从独立样本集合中构建批次的输入输出，并转换为PyTorch张量
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, context_size = contexts.shape
        neg_contexts = []
        # 对批次内样本分别进行负采样
        for i in range(batch_size):
            ns_dist = ns_dist.index_fill(0, contexts[i], .0)
            # index_fill根据给定索引填充张量的值，此处用于保持context不变
            neg_contexts.append(torch.multinomial(ns_dist, n_negatives * context_size, replacement=True))
            # 对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标
        neg_contexts = torch.stack(neg_contexts, dim=0)
        # stack用于拼接
        return words, contexts, neg_contexts
    
