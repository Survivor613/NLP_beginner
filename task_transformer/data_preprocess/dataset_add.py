# Transformer标准架构

import random
from data_preprocess.dataset_base import BaseDataset

class AddDataset(BaseDataset):
    def __init__(self, vocab, tokenizer):
        super().__init__(self)
        self.data = self._load_data()
        self.vocab = vocab
        self.tokenizer = tokenizer

    def _load_data(self):
        data = []
        for _ in range(1000):
            a_ones = random.randint(5, 9)  # a的个位：5-9
            # 确保b_ones ≥ (10 - a_ones)，且b_ones ≥ 1（避免0），同时右边界≥左边界
            min_b_ones = max(10 - a_ones, 1)  # 最小b_ones：保证a+b≥10且至少为1
            max_b_ones = 9  # 最大b_ones：9（个位最大数字）
            b_ones = random.randint(min_b_ones, max_b_ones)  # 此时范围一定有效
            a_tens = random.randint(1, 9)
            a = f"{a_tens}{a_ones}"
            b = f"{b_ones}"
            data.append(self._format(a, b))

        for _ in range(5000):
            # 生成3-5位随机数字
            a = str(random.randint(100, 99999))  # 3-5位
            b = str(random.randint(100, 99999))
            data.append(self._format(a, b))
        
        # 强化进位样本（3000x2=6000个，覆盖更多连续进位场景）
        for _ in range(3000):
            # 场景1：全9数字（如999+xxx，易连续进位）
            a_len = random.randint(3, 5)
            a = '9' * a_len
            b = str(random.randint(1, 99999))  # 1-5位
            data.append(self._format(a, b))
            
            # 场景2：低位进位传递（如1999+2，中间位连续进位）
            a = str(random.randint(100, 99999))
            b = '9' * random.randint(1, min(3, len(a)))  # b的长度≤3且≤a的长度
            data.append(self._format(a, b))
        
        return data

    def _format(self, a, b):
        # 关键修复：补前导零使a和b位数相同，再反转（确保个位对齐）
        max_digit = max(len(a), len(b))  # 取两个数字的最大位数
        a_padded = a.zfill(max_digit)  # 补前导零至max_digit位（如"123"→"0123"当max_digit=4）
        b_padded = b.zfill(max_digit)
        
        # 计算真实结果
        c = str(int(a) + int(b))
        # 结果补前导零至（max_digit+1）位（应对进位后位数+1的情况，如999+1=1000）
        c_padded = c.zfill(max_digit + 1)
        
        # 反转（个位在前）
        a_rev = a_padded[::-1]
        b_rev = b_padded[::-1]
        c_rev = c_padded[::-1]
        
        # 输入格式："a_rev+b_rev"，输出格式："<s>c_rev</s>"
        return (f"{a_rev}+{b_rev}", f"<s>{c_rev}</s>")
    
    def _process_item(self, src, tgt):
        src_tokens = self.tokenizer(src)
        src_ids = [self.vocab[token] for token in src_tokens]

        tgt_tokens = self.tokenizer(tgt)
        tgt_ids = [self.vocab[token] for token in tgt_tokens]

        return src_ids, tgt_ids