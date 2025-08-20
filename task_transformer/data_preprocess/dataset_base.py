import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, max_len=None):
        self.max_len = max_len
        self.data = []

    def _load_data(self):
        """子类必须实现：加载原始数据，返回 [(src, tgt), ...]"""
        raise NotImplementedError

    def _process_item(self, src, tgt, add_special_tokens=True):
        """子类覆盖：将 src/tgt 文本转为 id"""
        raise NotImplementedError

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return self._process_item(src, tgt)

    def __len__(self):
        return len(self.data)


def pad_sequences(sequences, padding_value=0):
    seqs = [seq.tolist() if isinstance(seq, torch.Tensor) else seq for seq in sequences]
    max_len = max(len(seq) for seq in seqs)
    seq_padded = [seq + [padding_value]*(max_len - len(seq)) for seq in seqs]
    return torch.tensor(seq_padded, dtype=torch.long)


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_seq, tgt_seq = zip(*batch)
    src_padded = pad_sequences(src_seq, padding_value=src_pad_idx)
    tgt_padded = pad_sequences(tgt_seq, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded
