import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.sublayer import MultiHeadAttention, PositionwiseFeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src_seq, enc_self_attn_mask=None):
        """ MHA + Add & Norm """
        attn_output, _ = self.self_attn(src_seq, src_seq, src_seq, mask=enc_self_attn_mask)
        src_seq = src_seq + self.dropout1(attn_output) # 残差连接
        src_seq = self.norm1(src_seq)

        """ FFN + Add & Norm """
        ffn_output = self.feed_forward(src_seq)
        src_seq = src_seq + self.dropout2(ffn_output)
        src_seq = self.norm2(src_seq)

        return src_seq
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt_seq, memory=None, dec_self_attn_mask=None, dec_cross_attn_mask=None):
        """ MHA + Add & Norm """
        attn_output, _ = self.self_attn(tgt_seq, tgt_seq, tgt_seq, mask=dec_self_attn_mask)
        tgt_seq = tgt_seq + self.dropout1(attn_output)
        tgt_seq = self.norm1(tgt_seq)

        if memory is not None: # 如果是Decoder-Only Transformer，则不需要Cross Attention层
            """ Cross Attention + Add & Norm """
            attn_output, _ = self.cross_attn(tgt_seq, memory, memory, mask=dec_cross_attn_mask)
            tgt_seq = tgt_seq + self.dropout2(attn_output)
            tgt_seq = self.norm2(tgt_seq)

        """ FFN + Add & Norm """
        ffn_output = self.feed_forward(tgt_seq)
        tgt_seq = tgt_seq + self.dropout3(ffn_output)
        tgt_seq = self.norm3(tgt_seq)

        return tgt_seq
    

class TransformerOutputLayer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super(TransformerOutputLayer, self).__init__()
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, x):
        """ 将输出映射到词汇表大小 """
        logits = self.projection(x)
        probabilities = F.softmax(logits, dim=-1)

        return logits, probabilities