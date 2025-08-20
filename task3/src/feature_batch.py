import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def data_split(data, test_rate = 0.3):
    train = list()
    test = list()
    for datum in data:
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)

    return train, test


class Random_embedding():
    def __init__(self, data, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data] 
        """item 即为一行,经过split('\t')分隔后,对该行中每个字符串(这里一行只有一个字符串)按照制表符分隔,形成双层列表,类似[['neutral', '( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )', '( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )', '(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))', '(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))', 'Two women are embracing while holding to go packages.', 'The sisters are hugging goodbye while holding to go packages after just eating lunch.', '4705552913.jpg#2', '4705552913.jpg#2r1n', 'neutral', 'entailment', 'neutral', 'neutral', 'neutral']]"""
        self.data = [[item[5], item[6], item[0]] for item in _data]
        """提取每个字符串的第6,第7,第1个信息,分别为句子A原句,句子B原句,最终分类结果,构成双层列表"""
        self.data.sort(key=lambda x:len(x[0].split()))
        """lambda表达式输入x输出A原句长度,sort()根据长度进行排序,方便后续padding"""
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]
        """将分类结果向量化"""
        self.test_y = [self.type_dict[term[2]] for term in self.test]
        self.train_s1_matrix = list() # 句子A矩阵
        self.test_s1_matrix = list()
        self.train_s2_matrix = list() # 句子B矩阵
        self.test_s2_matrix = list()
        self.longest = 0
    
    def get_words(self):
        pattern = '[A-Za-z\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s) # 根据正则表达式返回符合条件的无重复列表
                for word in words:
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words) + 1
        self.len_words = len(self.dict_words) + 1 # 加入''作为padding所需的特殊词

    def set2id(self, set, sent_id, matrix): # 辅助函数
        pattern = '[A-Za-z\']+'
        for term in set:
            s = term[sent_id]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            matrix.append(item)

    def get_id(self):
        self.set2id(self.train, 0, self.train_s1_matrix)
        self.set2id(self.train, 1, self.train_s2_matrix)
        self.set2id(self.test, 0, self.test_s1_matrix)
        self.set2id(self.test, 1, self.test_s2_matrix)
        

class Glove_embedding():
    def __init__(self, data, trained_dict, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data] 
        """item 即为一行,经过split('\t')分隔后,对该行中每个字符串(这里一行只有一个字符串)按照制表符分隔,形成双层列表,类似[['neutral', '( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )', '( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )', '(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))', '(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))', 'Two women are embracing while holding to go packages.', 'The sisters are hugging goodbye while holding to go packages after just eating lunch.', '4705552913.jpg#2', '4705552913.jpg#2r1n', 'neutral', 'entailment', 'neutral', 'neutral', 'neutral']]"""
        self.data = [[item[5], item[6], item[0]] for item in _data]
        """提取每个字符串的第6,第7,第1个信息,分别为句子A原句,句子B原句,最终分类结果,构成双层列表"""
        self.data.sort(key=lambda x:len(x[0].split()))
        """lambda表达式输入x输出A原句长度,sort()根据长度进行排序,方便后续padding"""
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]
        """将分类结果向量化"""
        self.test_y = [self.type_dict[term[2]] for term in self.test]
        self.train_s1_matrix = list() # 句子A矩阵
        self.test_s1_matrix = list()
        self.train_s2_matrix = list() # 句子B矩阵
        self.test_s2_matrix = list()
        self.longest = 0

        # 差异
        self.trained_dict = trained_dict
        self.embedding = list()

    def get_words(self):
        self.embedding.append([0]*50) # 为''预留的词向量表示(全0)
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s)
                for word in words:
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)+1
                        if word in self.trained_dict:
                            self.embedding.append(self.trained_dict[word])
                        else:
                            self.embedding.append([0] * 50)
        self.len_words = len(self.dict_words) + 1
    
    def set2id(self, set, sent_id, matrix): # 辅助函数
        pattern = '[A-Za-z\']+'
        for term in set:
            s = term[sent_id]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            matrix.append(item)

    def get_id(self):
        self.set2id(self.train, 0, self.train_s1_matrix)
        self.set2id(self.train, 1, self.train_s2_matrix)
        self.set2id(self.test, 0, self.test_s1_matrix)
        self.set2id(self.test, 1, self.test_s2_matrix)


class ClsDataset(Dataset):
    """文本分类数据集"""
    def __init__(self, sentence1, sentence2, relation):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.relation = relation

    def __getitem__(self, index):
        return self.sentence1[index], self.sentence2[index], self.relation[index]

    def __len__(self):
        return len(self.relation)

def collate_fn(batch_data):
    """自定义一个batch里面数据的组织方式"""
    sents1, sents2, labels = zip(*batch_data)
    sentences1 = [torch.LongTensor(sent) for sent in sents1]
    padded_sents1 = pad_sequence(sentences1, batch_first=True, padding_value=0)
    sentences2 = [torch.LongTensor(sent) for sent in sents2]
    padded_sents2 = pad_sequence(sentences2, batch_first=True, padding_value=0)
    return padded_sents1, padded_sents2, torch.LongTensor(labels)

def get_batch(x1, x2, y, batch_size):
    dataset = ClsDataset(x1, x2, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader