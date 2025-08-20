import random
from torch.utils.data import Dataset, DataLoader # 用DataLoader来批量取数据，而DataLoader仅仅能接受Dataset类型的数据
from torch.nn.utils.rnn import pad_sequence  # 用于对batch内的句子进行padding
import torch


def data_split(train_data, val_rate=0.3):
    """把数据按一定比例分成训练集和测试集"""
    train = list()
    val = list()
    for datum in train_data:
        if random.random() > val_rate:
            train.append(datum)
        else:
            val.append(datum)
    return train, val

class Random_embedding():  # 词向量还未生成
    """随机初始化"""
    def __init__(self, train_data, test_data, val_rate=0.1):
        self.dict_words = dict() # 词->id
        train_data.sort(key=lambda x: len(x[0].split())) # split取出第三项并以空格划分计算词数量，按照长度排序，短的在前，这样做可以避免后面一个batch内句子长短不一，导致padding过度
        self.train_data = train_data
        test_data.sort(key=lambda x: len(x[0].split()))
        self.test_data = test_data
        self.len_words = 0 # 词数量
        self.train, self.val = data_split(train_data, val_rate) # 训练集，测试集划分
        self.test = test_data
        self.train_y = [int(term[1]) for term in self.train] # 训练集类别
        self.val_y = [int(term[1]) for term in self.val] # 测试集类别
        self.test_y = [int(term[1]) for term in self.test]
        self.train_matrix = list() # 训练集词列表，叠成矩阵
        self.val_matrix = list() # 测试集词列表，叠成矩阵
        self.test_matrix = list()
        self.longest = 0; # 记录最长的词

    def get_words(self):
        for term in self.train_data:
            s = term[0] # 取出句子
            s = s.upper() # 转成大写(否则i,I等会被当成不同的词)
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1  # ID为0的预留给padding

        for term in self.test_data:
            s = term[0] # 取出句子
            s = s.upper() # 转成大写(否则i,I等会被当成不同的词)
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1  # ID为0的预留给padding

        self.len_words = len(self.dict_words) # 词数目，暂未包括padding

    def get_id(self): # 获取train_matrix和val_matrix，将句子转换成ID矩阵，便于后续随机向量初始化
        for term in self.train:
            s = term[0]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))  # 更新最长的词
            self.train_matrix.append(item)
        for term in self.val:
            s = term[0]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))  # 更新最长的词
            self.val_matrix.append(item)
        for term in self.test:
            s = term[0]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))  # 更新最长的词
            self.test_matrix.append(item)

        self.len_words += 1 # 加入padding



class Glove_embedding():  # 词向量也还未生成，但已经记录在embedding阵中
    """Glove初始化"""
    def __init__(self, train_data, test_data, trained_dict, d_model, val_rate=0.1):
        self.dict_words = dict()  # 词->id
        train_data.sort(key=lambda x: len(x[0].split()))  # 按句子长度排序，避免过度padding
        self.train_data = train_data
        self.test_data = test_data
        self.len_words = 0  # 词数量
        self.train, self.val = data_split(train_data, val_rate)  # 训练集/验证集划分
        self.test = test_data

        self.train_y = [int(term[1]) for term in self.train]  # 训练集类别
        self.val_y = [int(term[1]) for term in self.val]      # 验证集类别
        self.test_y = [int(term[1]) for term in self.test]    # 测试集类别

        self.train_matrix = list()  # 训练集ID序列
        self.val_matrix = list()    # 验证集ID序列
        self.test_matrix = list()   # 测试集ID序列
        self.longest = 0  # 记录最长的句子
        self.d_model = d_model

        # Glove embedding 部分
        self.trained_dict = trained_dict  # Glove预训练词向量
        self.embedding = list()  # 实际用到的词向量

    def get_words(self):
        self.embedding.append([0.0] * self.d_model)  # 先加入padding向量
        for term in self.train_data:
            s = term[0].upper()
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1  # ID=0 预留给padding
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])  # 加入预训练向量
                    else:
                        self.embedding.append([0.0] * self.d_model)

        for term in self.test_data:  # 确保测试集中词也被收录
            s = term[0].upper()
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:
                        self.embedding.append([0.0] * self.d_model)

        self.len_words = len(self.dict_words)  # 不含padding

    def get_id(self):  # 将句子转换为ID矩阵
        for term in self.train:
            s = term[0].upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_matrix.append(item)

        for term in self.val:
            s = term[0].upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.val_matrix.append(item)

        for term in self.test:
            s = term[0].upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_matrix.append(item)

        self.len_words += 1  # 加入padding


# 以下部分作用：实质为一个适配器，将文本数据（已转换为 ID 序列）组织成模型可直接训练的批次（batch）格式
class ClsDataset(Dataset):
    """自定义数据集类(pytorch基本功)"""
    def __init__(self, sentence, emotion):
        self.sentence = sentence  # 句子
        self.emotion = emotion  # 情感类别

    def __getitem__(self, item):
        return self.sentence[item], self.emotion[item]
    
    def __len__(self):
        return len(self.emotion)

def collate_fn(batch_data):
    """自定义数据集的内数据返回方式,并进行padding(pytorch基本功)"""
    sentences, emotions = zip(*batch_data)                                              # 解包
    sentences = [torch.LongTensor(sent) for sent in sentences]                          # 将句子转换为LongTensor类型
    emotions = torch.LongTensor(emotions)                                               #将情感类别转换为LongTensor类型
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)           # 对句子进行padding, batch_first使得矩阵每行是一个句子，符合直觉和任务需求
    return padded_sents, emotions

def get_batch(x, y, batch_size):
    """利用dataloader划分batch,获取一个batch的数据(pytorch基本功)"""
    # 迭代dataloader时，DataLoader会从dataset中取出batch_size个样本,并将batch作为参数传入collate_fn函数，并将函数返回值作为最终的批次数据，返回给用户
    dataset = ClsDataset(x, y)  # 创建数据集对象
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)  # 创建数据加载器，shuffle=False防止打乱顺序
    return dataloader