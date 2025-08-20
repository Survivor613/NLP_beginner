import torch
import torch.nn as nn
from transformer.layer import TransformerEncoderLayer, TransformerDecoderLayer, TransformerOutputLayer
from transformer.sublayer import get_enc_self_mask, get_dec_self_mask, get_dec_cross_mask

class Embedding(nn.Module):
    """ 输入:经过PreProcessing的词ID, 形状为(batch_size, seq_len),不足seq_len的部分用id=0进行padding
        输出: 形状为(batch_size, seq_len, d_model)的词向量"""
    def __init__(self, vocab_size, d_model, padding_idx=0):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        x_seq = self.embedding(x)  # 得到x_seq形状为(batch_size, seq_len, d_model)
        x_seq *= self.d_model ** 0.5  # nn.Embedding默认用均匀分布初始化,范围是[−sqrt(1/vocab_size), sqrt(1/vocab_size)],需要乘以sqrt(d_model)来平衡嵌入向量与位置编码的量级

        return x_seq


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._generate_positional_encoding()

    def _generate_positional_encoding(self):
        # 创造一个固定大小的矩阵(max_len, d_model),存放位置编码
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(-1)  # 形状为(max_len, 1)
        div_term = torch.exp(-1 * torch.arange(0, self.d_model, 2) * torch.log(torch.tensor(10000.0)) / self.d_model).unsqueeze(0)  # 形状为(1, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term) # dim 2i 利用广播机制
        pe[:, 1::2] = torch.cos(position * div_term) # dim 2i+1
        pe = pe.unsqueeze(0)  # 形状变为(1, max_len, d_model)
        self.register_buffer('pe', pe)  # 将pe注册为buffer,模型保存时会一并保存,但不会被优化器更新

        return pe

    def forward(self, x):
        """ x的形状为(batch_size, seq_len, d_model) """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # 位置编码与词向量相加,利用广播机制

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout, num_layers):
        super(TransformerEncoder, self).__init__()
        """ PyTorch 容器类 """
        self.layers = nn.ModuleList([TransformerEncoderLayer(n_head, d_model, d_hidden, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src_seq, enc_self_mask=None):
        """ src_seq形状为 (batch_size, seq_len, d_model) """
        output = src_seq
        for layer in self.layers:
            output = layer(output, enc_self_mask)

        return output
    

class TransformerDecoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout,  num_layers):
        super(TransformerDecoder, self).__init__()
        """ PyTorch 容器类 """
        self.layers = nn.ModuleList([TransformerDecoderLayer(n_head, d_model, d_hidden, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt_seq, memory=None, dec_self_mask=None, dec_cross_mask=None):
        """ tgt_seq形状为 (batch_size, seq_len, d_model) """
        output = tgt_seq
        for layer in self.layers:
            output = layer(output, memory, dec_self_mask, dec_cross_mask) # dec_self_mask是第一层的mask，dec_cross_mask是第二层的mask

        return output
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, n_head=8, d_model=512, d_hidden=2048, num_layers=6, dropout=0.1, src_pad_idx=0, tgt_pad_idx=0):
        super(Transformer, self).__init__()
        self.src_embedding = Embedding(src_vocab_size, d_model, src_pad_idx)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model, tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(n_head, d_model, d_hidden, dropout, num_layers)
        self.decoder = TransformerDecoder(n_head, d_model, d_hidden, dropout, num_layers)
        self.projector = TransformerOutputLayer(d_model, tgt_vocab_size)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src_id_seq, tgt_id_seq, decoder_only=False, enc_self_mask=None, dec_self_mask=None, dec_cross_mask=None):
        src_seq = self.src_embedding(src_id_seq)  # 得到src_seq形状为(batch_size, seq_len, d_model)
        tgt_seq = self.tgt_embedding(tgt_id_seq)  # 得到tgt_seq形状为(batch_size, seq_len, d_model)
        src_seq = self.positional_encoding(src_seq)  # 添加位置编码
        tgt_seq = self.positional_encoding(tgt_seq)

        if enc_self_mask is None:
            enc_self_mask = get_enc_self_mask(src_id_seq, self.src_pad_idx)  # 得到src的padding mask
        if dec_self_mask is None:
            dec_self_mask = get_dec_self_mask(tgt_id_seq, self.tgt_pad_idx)  # 得到tgt的padding mask
        if dec_cross_mask is None:
            dec_cross_mask = get_dec_cross_mask(src_id_seq, tgt_id_seq, self.src_pad_idx, self.tgt_pad_idx)  # 得到tgt对src的cross mask

        if decoder_only:
            memory = None  # 如果是Decoder-Only Transformer，则不需要编码器输出
        else:
            memory = self.encoder(src_seq, enc_self_mask=enc_self_mask)  # 编码器输出     
        decoder_output = self.decoder(tgt_seq, memory, dec_self_mask=dec_self_mask, dec_cross_mask=dec_cross_mask)  # 解码器输出
        logits, probabilities = self.projector(decoder_output)  # 投影到最终分类维度

        return logits, probabilities


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # ----------------- 超参数 -----------------
#     src_vocab_size = 10
#     tgt_vocab_size = 10
#     seq_len_src = 6
#     seq_len_tgt = 5
#     batch_size = 1  # 可视化单个样本
#     d_model = 16
#     padding_idx = 0

#     # ----------------- 随机输入 -----------------
#     src_id_seq = torch.randint(1, src_vocab_size, (batch_size, seq_len_src))
#     tgt_id_seq = torch.randint(1, tgt_vocab_size, (batch_size, seq_len_tgt))
#     src_id_seq[0, -1] = padding_idx
#     tgt_id_seq[0, -2:] = padding_idx

#     print("SRC IDs:", src_id_seq)
#     print("TGT IDs:", tgt_id_seq)

#     # ----------------- 构建模型 -----------------
#     model = Transformer(src_vocab_size, tgt_vocab_size, d_model=d_model, num_layers=1, padding_idx=padding_idx)

#     # ----------------- 生成 mask -----------------
#     enc_self_mask = get_enc_self_mask(src_id_seq, padding_idx)      # (B, 1, 1, S_src)
#     dec_self_mask = get_dec_self_mask(tgt_id_seq, padding_idx)      # (B, 1, T_tgt, T_tgt)
#     dec_cross_mask = get_dec_cross_mask(src_id_seq, tgt_id_seq, padding_idx)  # (B, 1, T_tgt, S_src)

#     # ----------------- 可视化函数 -----------------
#     def plot_mask(mask, title):
#         mask_np = mask[0,0].int().cpu().numpy()  # 取batch第0个样本，方便可视化
#         plt.imshow(mask_np, cmap='gray_r')  # 1表示屏蔽，0表示可访问
#         plt.title(title)
#         plt.colorbar()
#         plt.show()

#     plot_mask(enc_self_mask, "Encoder Self Mask")
#     plot_mask(dec_self_mask, "Decoder Self Mask (Padding + Causal)")
#     plot_mask(dec_cross_mask, "Decoder Cross Mask")

#     # ----------------- 前向传播 -----------------
#     logits, probs = model(src_id_seq, tgt_id_seq)
#     print("Logits shape:", logits.shape)
#     print("Probabilities shape:", probs.shape)

