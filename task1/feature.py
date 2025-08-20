import numpy as np
import random

def data_split(train_data, val_rate=0.3, max_item=1000):
    train = list()
    val = list()
    i = 0
    for datum in train_data:
        i += 1
        if random.random() > val_rate: # generate float number between 0 and 1
            train.append(datum)
        else:
            val.append(datum)
        if i > max_item:
            break
    return train, val

class Bag:
    """Bag of words"""
    def __init__(self, train_data, test_data, max_item):
        self.train_data = train_data[:max_item]
        self.test_data = test_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()     # words to word indexes
        self.len = 0                 # word amount
        self.train, self.val = data_split(self.train_data, val_rate=0.3, max_item=1000) # divide into train and val set
        self.test = self.test_data
        train_y = [int(term[1]) for term in self.train]   # train_y
        self.train_y = np.array(train_y)
        val_y = [int(term[1]) for term in self.val] # val_y
        self.val_y = np.array(val_y)
        test_y = [int(term[1]) for term in self.test]
        self.test_y = np.array(test_y)
        self.train_matrix = None
        self.val_matrix = None
        self.test_matrix = None
    
    def get_words(self):
        for term in self.train_data:
            s = term[0]
            s = s.upper() # turn into capital letter to ensure a word is spelled identically
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        for term in self.test_data:
            s = term[0]
            s = s.upper() # turn into capital letter to ensure a word is spelled identically
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.train_matrix = np.zeros((len(self.train), self.len))
        self.val_matrix = np.zeros((len(self.val), self.len))
        self.test_matrix = np.zeros((len(self.test), self.len))

    def get_matrix(self):
        for i in range(len(self.train)):
            s = self.train[i][0]
            words = s.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.dict_words[word]] = 1

        for i in range(len(self.val)):
            s = self.val[i][0]
            words = s.split()
            for word in words:
                word = word.upper()
                self.val_matrix[i][self.dict_words[word]] = 1

        for i in range(len(self.test)):
            s = self.test[i][0]
            words = s.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.dict_words[word]] = 1
                
                
class Gram:
    """N-gram"""
    def __init__(self, train_data, test_data, dimension=2, max_item=1000):
        self.train_data = train_data[:max_item]
        self.test_data = test_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()
        self.len = 0
        self.dimension = dimension
        self.train, self.val = data_split(self.train_data, val_rate=0.3, max_item=max_item)
        self.test = self.test_data

        train_y = [int(term[1]) for term in self.train]
        self.train_y = np.array(train_y)
        val_y = [int(term[1]) for term in self.val]
        self.val_y = np.array(val_y)
        test_y = [int(term[1]) for term in self.test]
        self.test_y = np.array(test_y)

        self.train_matrix = None
        self.val_matrix = None
        self.test_matrix = None
        
    def get_words(self):
        for d in range(1, self.dimension+1):  # get 1-gram, 2-gram ... n-gram
            for term in self.train_data:
                s = term[0]
                s = s.upper()
                words = s.split()
                for i in range(len(words)-d+1):
                    temp = '_'.join(words[i:i+d])
                    if temp not in self.dict_words:
                        self.dict_words[temp] = len(self.dict_words)

            for term in self.test_data:  # 确保 test 中的 n-gram 也收录
                s = term[0]
                s = s.upper()
                words = s.split()
                for i in range(len(words)-d+1):
                    temp = '_'.join(words[i:i+d])
                    if temp not in self.dict_words:
                        self.dict_words[temp] = len(self.dict_words)

        self.len = len(self.dict_words)
        self.train_matrix = np.zeros((len(self.train), self.len))
        self.val_matrix = np.zeros((len(self.val), self.len))
        self.test_matrix = np.zeros((len(self.test), self.len))
        
    def get_matrix(self):
        for d in range(1, self.dimension+1):
            # train
            for i in range(len(self.train)):
                s = self.train[i][0].upper()
                words = s.split()
                for j in range(len(words)-d+1):
                    temp = '_'.join(words[j:j+d])
                    self.train_matrix[i][self.dict_words[temp]] = 1

            # val
            for i in range(len(self.val)):
                s = self.val[i][0].upper()
                words = s.split()
                for j in range(len(words)-d+1):
                    temp = '_'.join(words[j:j+d])
                    self.val_matrix[i][self.dict_words[temp]] = 1

            # test
            for i in range(len(self.test)):
                s = self.test[i][0].upper()
                words = s.split()
                for j in range(len(words)-d+1):
                    temp = '_'.join(words[j:j+d])
                    self.test_matrix[i][self.dict_words[temp]] = 1
