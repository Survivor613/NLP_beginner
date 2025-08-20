import torch
import torch.nn as nn
import torch.nn.functional as F

def get_enc_self_mask(src_id_seq, padding_idx):
    src_pad_seq = (src_id_seq != padding_idx).unsqueeze(-1).float()  # 利用广播机制(将padding_idx标量广播成向量,并逐元素比较), pad_seq形状为(batch_size, seq_len, 1), 即padding位置为0, 非padding位置为1
    enc_self_mask = torch.matmul(src_pad_seq, src_pad_seq.transpose(-1, -2))  # 得到pad_mask形状为(batch_size, seq_len, seq_len), 用于mask padding部分

    return enc_self_mask.bool()

def get_dec_cross_mask(src_id_seq, tgt_id_seq, src_pad_idx, tgt_pad_idx):
    src_pad_seq = (src_id_seq != src_pad_idx).unsqueeze(-1).float() # 转换成float类型,因为bmm不支持bool类型的矩阵乘法
    tgt_pad_seq = (tgt_id_seq != tgt_pad_idx).unsqueeze(-1).float()
    dec_cross_mask = torch.matmul(tgt_pad_seq, src_pad_seq.transpose(-1, -2))

    return dec_cross_mask.bool() 

def get_dec_self_mask(tgt_id_seq, padding_idx):
    tgt_seq_len = tgt_id_seq.size(-1)
    subsequent_mask = torch.tril(torch.ones((1, tgt_seq_len, tgt_seq_len), dtype=torch.bool, device=tgt_id_seq.device)) # 下三角矩阵,形状为(1, tgt_seq_len, tgt_seq_len)
    pad_mask = get_enc_self_mask(tgt_id_seq, padding_idx)  # 借用get_enc_self_mask函数, 得到tgt的padding mask
    dec_self_mask = pad_mask & subsequent_mask  # 逻辑与运算,使得均为1的位置为1,其余位置为0,同时利用广播机制使tril自动匹配batch_size维度

    return dec_self_mask.bool()

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.W_q = nn.Linear(d_model, n_head * self.d_k)
        self.W_k = nn.Linear(d_model, n_head * self.d_k)
        self.W_v = nn.Linear(d_model, n_head * self.d_v)
        self.W_o = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5  # 缩放因子,因为Q,K形状会变成(seq_len, batch_size * n_head, d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1) # key和value长度一定相同,但key和question长度可能不同

        # 乘以W_q,W_k,W_v矩阵,得到QKV矩阵
        query = self.W_q(query).view(batch_size, q_len, self.n_head, self.d_k)
        query = query.view(batch_size * self.n_head, q_len, self.d_k)
        key = self.W_k(key).view(batch_size, k_len, self.n_head, self.d_k)
        key = key.view(batch_size * self.n_head, k_len, self.d_k)
        value = self.W_v(value).view(batch_size, k_len, self.n_head, self.d_v)
        value = value.view(batch_size * self.n_head, k_len, self.d_v)

        # 计算注意力分数
        scores = (torch.matmul(query, key.transpose(-1, -2)) / self.scale)  # QKT/sqrt(dk), 形状为(batch_size * n_head, q_len, k_len)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)  # 形状变为(batch_size, n_head, q_len, k_len)
        mask = mask.view(batch_size * self.n_head, q_len, k_len)  # 形状变为(batch_size * self.n_head, q_len, k_len)
        scores = scores.masked_fill(~mask, -1e4) # ~mask指对mask取反,因为mask中非padding位置为True,而masked_fill是将True位置的值替换
        attn = torch.softmax(scores, dim=-1) # 形状仍然为(seq_len, batch_size * n_head, seq_len)
        attn = self.dropout(attn) # dropout
        output = torch.matmul(attn, value)  # Attention*Value, 形状为(batch_size * n_head, seq_len, d_v)
        output = output.view(batch_size, q_len, self.n_head * self.d_v)  # 形状变为(batch_size, q_len, n_head * d_v),即(batch_size, q_len, d_model)
        output = self.W_o(output)  # 乘以W_o矩阵, 进一步融合信息, 形状不变

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        """ 结构: Linear -> ReLU -> Dropout -> Linear """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    



# 测试FFN层代码
if __name__ == "__main__":
    # 超参数设置（与原论文保持一致）
    d_model = 512    # 模型特征维度
    d_ff = 2048      # 前馈网络中间层维度
    dropout = 0.1    # Dropout概率
    
    # 实例化位置前馈网络
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 生成随机输入（seq_len=10, batch_size=32, d_model=512）
    x = torch.randn(10, 32, d_model)
    
    # 前向传播
    output = ffn(x)
    
    # 验证输出形状是否正确
    assert output.shape == x.shape, f"输出形状错误！预期 {x.shape}，实际 {output.shape}"
    print(f"测试通过！输入形状: {x.shape}, 输出形状: {output.shape}")
