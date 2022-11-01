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

    
