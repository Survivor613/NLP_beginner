import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有GPU可用

class MY_CNN(nn.Module):
    """自定义CNN模型"""
    def __init__(self, len_feature, len_words, longest, typenum=5, weight=None, drop_out=0.1):
        super(MY_CNN, self).__init__()
        self.len_feature = len_feature # d的大小
        self.len_words = len_words # 词数目
        self.longest = longest # 最长句子的词数量
        self.drop_out = nn.Dropout(drop_out) # Dropout层

        # embedding层
        if weight is None: # 随机初始化
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=self.len_words, embedding_dim=self.len_feature, _weight=x)
        else: # Glove初始化
            self.embedding = nn.Embedding(num_embeddings=self.len_words, embedding_dim=self.len_feature, _weight=weight)

        # 卷积层 + 激活层
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=longest, kernel_size=(2, self.len_feature), padding=(1,0)), nn.ReLU())  # 一个卷积层接着一个RELU层
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=longest, kernel_size=(3, self.len_feature), padding=(1,0)), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=longest, kernel_size=(4, self.len_feature), padding=(2,0)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=longest, kernel_size=(5, self.len_feature), padding=(2,0)), nn.ReLU())  # 得到4个longest长度的向量，拼在一起成为4xlongest长度的向量

        # 全连接层
        self.fc = nn.Linear(4 * longest, typenum)

        # 将模型移动到GPU或CPU
        self.to(device)

    def forward(self, x):
        # embedding层
        """x为数据，维度为(batch_size, longest) """
        x = x.long() # 转换为LongTensor类型
        """经过embedding后，维度变为(batch_size, 1, longest, len_feature)"""
        output = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.len_feature)  # x.shape[0]获取张量x的第一个维度
        output = self.drop_out(output)  # Dropout层

        # 卷积层 + 激活层
        """经过卷积层后，维度变为(batch_size, l_l(out_channels), longest(+1), 1)"""
        conv1 = self.conv1(output).squeeze(3)
        conv1 = self.drop_out(conv1)
        conv2 = self.conv2(output).squeeze(3)
        conv2 = self.drop_out(conv2)
        conv3 = self.conv3(output).squeeze(3)
        conv3 = self.drop_out(conv3)
        conv4 = self.conv4(output).squeeze(3) # squeeze(3)去掉最后一个维度,维度变为(batch_size, l_l, len_feature)
        conv4 = self.drop_out(conv4)

        # 池化层（池化+拼接）
        """经过池化层后，维度变为(batch_size, l_l(out_channels), 1)"""
        pool1 = F.max_pool1d(conv1, conv1.shape[2])  # 对longest(+1)维度进行最大池化,维度变为(batch_size, l_l(=longest), 1)
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        pool4 = F.max_pool1d(conv4, conv4.shape[2])
        """拼接池化后的结果，维度变为(batch_size, 4 * longest, 1)"""
        pool = torch.cat((pool1, pool2, pool3, pool4), dim=1).squeeze(2)  # 在维度1上拼接，并squeeze(2)去掉最后一个维度，维度变为(batch_size, 4 * l_l)

        pool = self.drop_out(pool)

        # 全连接层
        """经过全连接层后，维度变为(batch_size, typenum), 通过乘以权重矩阵W, 维度为[ (4*l_l), typenum ]"""
        output = self.fc(pool)  # 全连接层, 同时处理batch_size个句子，输入层为（4*l_l）维列向量,通过线性变换得到y_hat(维度为typenum x 1)

        # 注:softmax已经在CrossEntropyLoss损失函数中实现了
        # 原因在于CrossEntropyLoss损失函数中连用Softmax + NLLLoss(Negative Log Likelihood Loss，负对数似然估计)可以使用LogSumExp trick,使得数值不易溢出
        # 具体请参考豆包
        # 故这里不需要再加一层softmax，只需要在最后进行预测时补一层softmax

        return output  # 返回预测结果y_hat矩阵，维度为(batch_size, typenum)

class MY_RNN(nn.Module):
    """自定义RNN模型"""
    def __init__(self, len_feature, len_hidden, len_words, typenum=5, weight=None, layer=1, nonlinearity='tanh', batch_first=True, drop_out=0.1): # 增添len_hidden参数，表示RNN的hidden_state的维度
        super(MY_RNN, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.drop_out = nn.Dropout(drop_out)

        # embedding层
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=self.len_words, embedding_dim=self.len_feature, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.len_words, embedding_dim=self.len_feature, _weight=weight)

        # RNN层
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity, batch_first=batch_first, dropout=drop_out)

        # 全连接层
        self.fc = nn.Linear(len_hidden, typenum)

        # 将模型移动到GPU或CPU
        self.to(device)

    def forward(self, x):
        """x为数据,维度为(batch_size, length),其中length为句子长度,但在RNN中句子长度不影响流程,故没有作为参数传入"""
        x = x.long()  # 转换为LongTensor类型
        batch_size = x.shape[0]  # 获取batch_size

        # embedding层
        """经过embedding后,维度变为(batch_size, length, len_feature)"""
        output = self.embedding(x)
        output = self.drop_out(output)  # Dropout层

        # RNN层
        """初始化RNN的初始隐藏状态h0为全0张量,维度为(layer, batch_size, len_hidden)"""
        h0 = torch.zeros(self.layer, batch_size, self.len_hidden, device=device) # 在GPU上创建h0
        """hn为RNN的最后一个隐藏状态,维度为(1, batch_size, len_hidden)"""
        _, hn = self.rnn(output, h0)

        hn = hn[-1, :, :]
        hn = self.drop_out(hn)

        # 全连接层
        """经过全连接层后,维度变为(1, batch_size, typenum), 通过乘以权重矩阵W, 维度为[ len_hidden, typenum ]"""
        output = self.fc(hn).squeeze(0)  # squeeze(0)去掉第一个维度,维度变为(batch_size, typenum)

        return output  # 返回预测结果y_hat矩阵，维度为(batch_size, typenum)

        