import torch
import torch.nn as nn


class Input_Encoding(nn.Module):
    """embedding + BiLSTM"""
    def __init__(self, len_feature, len_hidden, len_words, longest, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.longest = longest
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words,len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words,embedding_dim=len_feature, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words,embedding_dim=len_feature, _weight=weight)
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first, bidirectional=True)
        """batch_first=True条件下,输入张量的形状为 [batch_size, sequence_length, feature_dim], bidrectional=True表明为BiLSTM"""
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class Local_Inference_Modeling(nn.Module):
    """ 跨语句注意力模块 """
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1)
        """这里是三维, dim分别为0,1,2, dim=1代表sequence_length维度,即纵向softmax"""
        self.softmax_2 = nn.Softmax(dim=2)
        """dim=2即为横向softmax"""

    def forward(self, a_bar, b_bar):
        e = torch.matmul(a_bar, b_bar.transpose(1, 2))
        """matmul,即matrix_multiply(矩阵乘法),b_bar原本为[batch_size, sequence_length, feature_dim],经过dim1和2的置换后,变为[batch_size, feature_dim, sequence_length]"""
        """即为 E = A * BT """
    
        a_tilde = self.softmax_2(e) # tilde表示波浪线符号
        a_tilde = a_tilde.bmm(b_bar)
        """bmm,即batch_matrix_multiply,批量矩阵乘法,bmm必须三维张量,且后两维维度匹配(符合乘法规则),而若要transpose则需使用上文的torch.matmul()"""
        """即为 a~ = softmax_row(E) * B """
        b_tilde = self.softmax_1(e)
        b_tilde = b_tilde.transpose(1, 2).bmm(a_bar)
        """即为 b~ = (softmax_col(E))T * A """

        m_a = torch.cat([a_bar, a_tilde, a_bar-a_tilde, a_bar * a_tilde], dim=-1) # cat即为concat(拼接)
        m_b = torch.cat([b_bar, b_tilde, b_bar-b_tilde, b_bar * b_tilde], dim=-1)
        """ 四个矩阵参数均为 [batch_size, sequence_length, 2 * len_hidden], 在最后一维拼接变为8 * len_hidden"""

        return m_a, m_b
    

class Inference_Composition(nn.Module):
    """ 全连接 + Dropout + BiLSTM """
    def __init__(self, len_feature, len_hidden_m, len_hidden, layer=1, batch_first=True,drop_out=0.5):
        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(len_hidden_m, len_feature)
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first, bidirectional=True)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, x):
        x = self.linear(x) # 转化为[batch_size, sequence_length, len_feature]
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # 转化为[batch_size, sequence_length, 2 * len_hidden]

        return x
    

class Prediction(nn.Module):
    """ Pooling + 拼接 + Dropout + MLP(Multi-Layer Perceptron,等价于多隐藏层的全连接神经网络) """
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5): # len_v为输入向量最后一维大小,len_mid为经过1层全连接层之后的长度,type_num为最后的分类数,对应'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(), nn.Linear(len_mid, type_num))

    def forward(self, a, b):
        """ 先进行pooling+拼接, 在接上MLP层 """
        v_a_avg = a.sum(1)/a.shape[1] # 1代表第二维度,即沿着第二维度(顺着列)求平均,(若沿行求平均,会破坏特征表达)
        v_a_max = a.max(1)[0] # max返回最大值和索引,这里的[0]代表只关注最大值,不关注索引
        v_b_avg = b.sum(1)/b.shape[1]
        v_b_max = b.max(1)[0]

        output = torch.cat((v_a_avg, v_a_max, v_b_avg, v_b_max), dim=-1)

        return self.mlp(output)


class ESIM(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words, longest, type_num=4, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words = len_words
        self.longest = longest
        """ 四层 """
        self.input_encoding = Input_Encoding(len_feature, len_hidden, len_words, longest, weight, layer, batch_first, drop_out)
        self.local_inference_modeling = Local_Inference_Modeling()
        self.Inference_composition = Inference_Composition(len_feature, 8*len_hidden, len_hidden, layer, batch_first, drop_out)
        self.prediction = Prediction(8*len_hidden, len_hidden, type_num, drop_out)
    
    def forward(self, a, b):
        a_bar = self.input_encoding(a)
        b_bar = self.input_encoding(b)

        m_a, m_b = self.local_inference_modeling(a_bar, b_bar)

        v_a = self.Inference_composition(m_a)
        v_b = self.Inference_composition(m_b)

        output = self.prediction(v_a, v_b)

        return output