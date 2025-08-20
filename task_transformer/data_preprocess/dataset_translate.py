from data_preprocess.dataset_base import BaseDataset

class TranslateDataset(BaseDataset):
    def __init__(self, data_iter, de_vocab, en_vocab, de_tokenizer, en_tokenizer, max_len=None):
        super().__init__(max_len=max_len)
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer
        self.raw_data_iter = list(data_iter)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for example in self.raw_data_iter:
            if isinstance(example, tuple):
                src_raw, tgt_raw = example
            else:
                src_raw, tgt_raw = example.src, example.trg

            if isinstance(src_raw, list):
                src_raw = " ".join(src_raw)
            if isinstance(tgt_raw, list):
                tgt_raw = " ".join(tgt_raw)

            data.append((src_raw, tgt_raw))
        return data

    def _process_item(self, src, tgt, add_special_tokens=True):
        # ---------------- src ----------------
        src_tokens = ["<s>"] + self.de_tokenizer(src) + ["</s>"]
        if self.max_len:
            max_mid_len = self.max_len - 2
            mid_tokens = src_tokens[1:-1][:max_mid_len]
            src_tokens = ["<s>"] + mid_tokens + ["</s>"]
        src_ids = [self.de_vocab[token] for token in src_tokens]

        # ---------------- tgt ----------------
        tgt_tokens = ["<s>"] + self.en_tokenizer(tgt) + ["</s>"]
        if self.max_len:
            max_mid_len = self.max_len - 2
            mid_tokens = tgt_tokens[1:-1][:max_mid_len]
            tgt_tokens = ["<s>"] + mid_tokens + ["</s>"]
        tgt_ids = [self.en_vocab[token] for token in tgt_tokens]

        return src_ids, tgt_ids


