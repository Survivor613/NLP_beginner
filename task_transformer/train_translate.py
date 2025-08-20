import torchtext
torchtext.disable_torchtext_deprecation_warning()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import random

from transformer.model import Transformer
from transformer.sublayer import get_enc_self_mask, get_dec_self_mask, get_dec_cross_mask
from data_preprocess.dataset_translate import TranslateDataset
from data_preprocess.dataset_base import collate_fn
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# ----------------- 配置 -----------------
class Config:
    batch_size = 128
    lr = 2e-4
    num_epochs = 30
    max_len = 40
    d_model = 256
    n_head = 8
    d_hidden = 1024
    num_layers = 8
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_steps = 4000
    grad_clip = 1.0
    patience = 5
    src_pad_idx = None
    tgt_pad_idx = None

# ----------------- 学习率调度 -----------------
def get_lr(step, d_model, warmup_steps=4000):
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

# ----------------- 数据处理 -----------------
def prepare_vocab_and_dataset(config: Config):
    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    def yield_tokens(data_iter, tokenizer, index):
        for example in data_iter:
            text = example[index] if isinstance(example, tuple) else getattr(example, 'src' if index==0 else 'trg')
            if isinstance(text, list):
                text = " ".join(text)
            yield tokenizer(text)

    train_iter_list = list(Multi30k(split='train', language_pair=('de','en')))
    specials = ["<unk>", "<pad>", "<s>", "</s>"]
    de_vocab = build_vocab_from_iterator(yield_tokens(train_iter_list, de_tokenizer, 0), specials=specials)
    en_vocab = build_vocab_from_iterator(yield_tokens(train_iter_list, en_tokenizer, 1), specials=specials)
    de_vocab.set_default_index(de_vocab["<unk>"])
    en_vocab.set_default_index(en_vocab["<unk>"])

    src_pad_idx = de_vocab["<pad>"]
    tgt_pad_idx = en_vocab["<pad>"]
    config.src_pad_idx = src_pad_idx
    config.tgt_pad_idx = tgt_pad_idx

    full_dataset = TranslateDataset(train_iter_list, de_vocab, en_vocab, de_tokenizer, en_tokenizer, config.max_len)
    dataset = list(full_dataset)
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.9)
    train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx),
        num_workers=2,
        pin_memory=True
    )
    return de_vocab, en_vocab, train_loader, val_loader, val_dataset, src_pad_idx, tgt_pad_idx

# ----------------- 训练 -----------------
def train_one_epoch(model, dataloader, criterion, optimizer, config, scaler, global_step, tgt_pad_idx, visualize_mask_batches=2):
    """
    visualize_mask_batches: 打印前几个 batch 的 mask，便于调试
    """
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(config.device), tgt.to(config.device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, _ = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        lr = get_lr(global_step, config.d_model, config.warmup_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 统计准确率
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = tgt_output != tgt_pad_idx
            total_correct += (pred == tgt_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

        total_loss += loss.item()

        # # ----------------- mask 可视化 -----------------
        # if batch_idx < visualize_mask_batches:
        #     enc_self_mask = get_enc_self_mask(src, model.padding_idx)    # (batch, src_len, src_len)
        #     dec_self_mask = get_dec_self_mask(tgt_input, model.padding_idx)  # (batch, tgt_len, tgt_len)
        #     dec_cross_mask = get_dec_cross_mask(src, tgt_input, model.padding_idx)  # (batch, tgt_len, src_len)

        #     batch = 0  # 仅打印 batch 0
        #     print(f"\nBatch {batch_idx+1} mask visualization (Batch {batch}):")
        #     print("Encoder Self Mask (seq_len x seq_len):\n", enc_self_mask[batch].int())
        #     print("Decoder Self Mask (Padding + Causal, tgt_len x tgt_len):\n", dec_self_mask[batch].int())
        #     print("Decoder Cross Mask (tgt_len x src_len):\n", dec_cross_mask[batch].int())

    avg_loss = total_loss / len(dataloader)
    acc = total_correct / total_tokens if total_tokens else 0.0
    return avg_loss, acc, global_step


# ----------------- Teacher-forcing 验证 -----------------
def evaluate_teacher_forcing(model, dataloader, config, de_vocab, en_vocab, tgt_pad_idx, max_samples=5):
    model.eval()
    total_correct, total_tokens = 0, 0
    samples = []

    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(config.device), tgt.to(config.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            logits, _ = model(src, tgt_input)

            pred = logits.argmax(dim=-1)
            mask = tgt_output != tgt_pad_idx
            total_correct += (pred == tgt_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            # 只收集部分样本
            for j in range(len(src)):
                if len(samples) < max_samples:
                    src_text = " ".join([de_vocab.get_itos()[id] for id in src[j].tolist() if id not in [de_vocab["<pad>"], de_vocab["<s>"], de_vocab["</s>"]]])
                    tgt_true = " ".join([en_vocab.get_itos()[id] for id in tgt[j].tolist() if id not in [tgt_pad_idx, en_vocab["<s>"], en_vocab["</s>"]]])
                    tgt_pred = " ".join([en_vocab.get_itos()[id] for id in pred[j].tolist() if id not in [tgt_pad_idx, en_vocab["<s>"], en_vocab["</s>"]]])
                    samples.append((src_text, tgt_true, tgt_pred))

    acc = total_correct / total_tokens if total_tokens else 0.0
    return acc, samples

# # ----------------- 自回归翻译 -----------------
# def translate_autoregressive(model, src_tensor, tgt_vocab, config, max_len=40):
#     model.eval()
#     src_tensor = src_tensor.to(config.device)
#     with torch.no_grad():
#         tgt_indices = torch.tensor([[tgt_vocab["<s>"]]], device=config.device)
#         for _ in range(max_len):
#             logits, _ = model(src_tensor, tgt_indices)
#             next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
#             tgt_indices = torch.cat([tgt_indices, next_token], dim=1)
#             if next_token.item() == tgt_vocab["</s>"]:
#                 break
#     # 去掉<s>和</s>
#     tokens = [tgt_vocab.get_itos()[idx] for idx in tgt_indices[0].tolist() if idx not in [tgt_vocab["<s>"], tgt_vocab["</s>"], tgt_vocab["<pad>"]]]
#     return " ".join(tokens)

def translate_autoregressive(model, src_tensor, tgt_vocab, config, max_len=40, visualize_mask_steps=2):
    """
    visualize_mask_steps: 打印前几个时间步的 mask
    """
    model.eval()
    src_tensor = src_tensor.to(config.device)
    with torch.no_grad():
        tgt_indices = torch.tensor([[tgt_vocab["<s>"]]], device=config.device)
        for step in range(max_len):
            logits, _ = model(src_tensor, tgt_indices)

            # # ----------------- mask 可视化 -----------------
            # if step < visualize_mask_steps:
            #     enc_self_mask = get_enc_self_mask(src_tensor, model.padding_idx)
            #     dec_self_mask = get_dec_self_mask(tgt_indices, model.padding_idx)
            #     dec_cross_mask = get_dec_cross_mask(src_tensor, tgt_indices, model.padding_idx)
            #     print(f"\nAR Step {step+1} mask visualization:")
            #     print("Encoder Self Mask (seq_len x seq_len):\n", enc_self_mask[0].int())
            #     print("Decoder Self Mask (tgt_len x tgt_len):\n", dec_self_mask[0].int())
            #     print("Decoder Cross Mask (tgt_len x src_len):\n", dec_cross_mask[0].int())

            # 自回归生成下一个 token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_indices = torch.cat([tgt_indices, next_token], dim=1)

            if next_token.item() == tgt_vocab["</s>"]:
                break

    # 去掉 <s> 和 </s>
    tokens = [tgt_vocab.get_itos()[idx] for idx in tgt_indices[0].tolist()
              if idx not in [tgt_vocab["<s>"], tgt_vocab["</s>"], tgt_vocab["<pad>"]]]
    return " ".join(tokens)


# ----------------- 主函数 -----------------
def main():
    config = Config()
    de_vocab, en_vocab, train_loader, val_loader, val_dataset, src_pad_idx, tgt_pad_idx = prepare_vocab_and_dataset(config)
    print(src_pad_idx, tgt_pad_idx)

    model = Transformer(
        src_vocab_size=len(de_vocab),
        tgt_vocab_size=len(en_vocab),
        n_head=config.n_head,
        d_model=config.d_model,
        d_hidden=config.d_hidden,
        num_layers=config.num_layers,
        dropout=config.dropout,
        src_pad_idx=config.src_pad_idx,
        tgt_pad_idx=config.tgt_pad_idx
    ).to(config.device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    global_step = 0

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, config, scaler, global_step, tgt_pad_idx, visualize_mask_batches=2
        )
        val_acc, val_samples = evaluate_teacher_forcing(
            model, val_loader, config, de_vocab, en_vocab, tgt_pad_idx
        )
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, TF Acc={train_acc:.4f}, Val TF Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Teacher-forcing 示例
    print("\n--- Teacher-forcing 示例 ---")
    for src, true, pred in val_samples:
        print(f"输入: {src}\n真实: {true}\n预测: {pred}\n")

    # 自回归示例
    print("\n--- 自回归翻译示例 ---")
    for i in range(min(5, len(val_dataset))):
        src_tensor, tgt_tensor = val_dataset[i]
        src_tensor = torch.tensor(src_tensor, dtype=torch.long).unsqueeze(0)
        pred_text = translate_autoregressive(model, src_tensor, en_vocab, config, max_len=config.max_len)
        src_text = " ".join([de_vocab.get_itos()[id] for id in src_tensor[0].tolist()])
        tgt_true = " ".join([en_vocab.get_itos()[id] for id in tgt_tensor])
        print(f"输入: {src_text}\n真实: {tgt_true}\n预测: {pred_text}\n")

if __name__ == "__main__":
    main()
