import numpy as np
import random

class Softmax:
    """softmax regression"""
    def __init__(self, sample, typenum, feature):
        self.sample = sample #len(bag.train)  数据条数
        self.typenum = typenum # 情感种类 -2,-1,0,1,2
        self.feature = feature # bag.len / len(dict_words) 单词数
        self.W = np.random.randn(feature, typenum)
        
    def softmax_calculation(self, x): # 返回列向量
        """x is a vector and get the softmax vector"""
        exp = np.exp(x - np.max(x)) # delete the max num without affecting the softmax value to prevent overstepping(越界)
        return exp/exp.sum() # return a softmax (colomn) vector
    
    def softmax_all(self, wtx): # wtx means x.dot(W)/WTx 返回行向量
        """wtx is a matrix and calculate softmax value row by row"""
        wtx -= np.max(wtx, axis=1, keepdims = True) # manipulate by row(axis=1) and np.max returns a column vector
        wtx = np.exp(wtx)
        wtx = wtx / np.sum(wtx, axis=1, keepdims = True)
        return wtx
        
    def change_y(self, y):
        """convert typenum of emotion into a one-hot vector"""
        ans = np.array([0] * self.typenum)
        ans[y] = 1  # turn one place into 1(one-hot)
        return ans.reshape(-1, 1) # -1: auto 1: column num, return a column vector

    def change_y_all(self, y): # 返回列向量
        """convert bag.train_y into a matrix"""
        ans = np.zeros((self.typenum, len(y)))
        for i in range(len(y)):
            ans[y[i]][i] = 1
        return ans  # a (typenum x len(y)) matrix
    
    def prediction(self, x):
        prob = self.softmax_all(x.dot(self.W))   # get softmax(WTx)
        return prob.argmax(axis=1)  # argmax by row and get index by colomn (0,1,2,3,4 representing 5 emotion types)
    
    def correct_rate(self, train, train_y, val, val_y, test, test_y):
        """calculate the accuracy of train and val set"""
        # train set
        pred_train = self.prediction(train)
        train_correct = sum(train_y[i] == pred_train[i] for i in range(len(train))) / len(train)
        # val set
        pred_val = self.prediction(val)
        val_correct = sum(val_y[i] == pred_val[i] for i in range(len(val))) / len(val)

        pred_test = self.prediction(test)
        test_correct = sum(test_y[i] == pred_test[i] for i in range(len(test))) / len(test)

        print(f"train_correctness:{train_correct}, val_correctness:{val_correct}, test_correctness:{test_correct}")
        return train_correct, val_correct, test_correct
    
    def regression(self, x, y, alpha, times, strategy="mini", mini_size=100): 
        # x : bag.train_matrix, y : bag.train_y, times: loop times
        """Softmax regression"""
        if self.sample != len(x) or self.sample != len(y):
            raise Exception("Sample size does not match!")    # is these two lines necessary? Aren't they determined by one thing?
        
        # shuffle (random gradient)
        if strategy == "shuffle":
            for i in range(times):
                k = random.randint(0, self.sample-1)
                y_hat = self.softmax_calculation(self.W.T.dot(x[k].reshape(-1,1)))
                increment = x[k].reshape(-1,1).dot((self.change_y(int(y[k])) - y_hat).T) # column vector x row vector
                self.W += alpha * increment
        
        # batch
        elif strategy == "batch":
            for i in range(times):
                y_hat = self.softmax_all(x.dot(self.W))
                increment = x.T.dot(self.change_y_all(y).T - y_hat) # feature x typenum matrix
                self.W += alpha * (increment / self.sample)
                
        # mini-batch        
        elif strategy == "mini":
            for i in range(times):
                k_choice = random.sample(range(self.sample), mini_size)
                x_random = x[k_choice]
                y_random = y[k_choice]
                y_hat = self.softmax_all(x_random.dot(self.W))
                increment = x_random.T.dot(self.change_y_all(y_random).T - y_hat)
                # size of x_random: len(k_choice) x feature
                # size of   (self.change_y_all(y_random).T - y_hat): len(k_choice) x typenum
                self.W += alpha * (increment / mini_size)
                    
        else:
            raise Exception("Unknown strategy")
        