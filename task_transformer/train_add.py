from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from tqdm import tqdm
from transformer.model import Transformer
from data_preprocess.dataset_base import BaseDataset, collate_fn
from data_preprocess.dataset_add import AddDataset

# 配置参数（优化词汇表和max_len）
class Config:
    def __init__(self):
        # 词汇表精简：移除冗余的'<', '>'，仅保留完整标记
        self.vocab = {
            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, 
            '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
            '+': 11, '<s>': 12, '</s>': 13,  # 仅保留必要标记
            '<pad>': 0                       # 填充标记
        }
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 模型参数（保持增强后的容量）
        self.d_model = 64
        self.n_head = 8
        self.num_layers = 6
        self.d_hidden = 256  # 增大FFN维度，增强拟合能力
        self.dropout = 0.15  # 适度提高dropout防过拟合
        
        # 训练参数
        self.batch_size = 32
        self.epochs = 40  # 延长训练轮次
        self.lr = 3e-4
        self.patience = 15  # 放宽早停条件
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 8  # 合理化max_len：5位数字反转后+加号（1）+5位反转=11，输出最长6位反转+2标记=8，总max_len=14足够
        self.save_path = 'add_model_aligned.pt'


# 分词器：适配精简后的词汇表
def tokenize(text, vocab):
    tokens = []
    i = 0
    while i < len(text):
        # 处理特殊标记
        if text.startswith('<s>', i):
            tokens.append('<s>')
            i += 3
        elif text.startswith('</s>', i):
            tokens.append('</s>')
            i += 4
        else:
            # 单个字符（数字或+）
            tokens.append(text[i])
            i += 1
    return tokens

# # 训练/验证函数（保持不变）
# def train_step(model, dataloader, criterion, optimizer, config):
#     model.train()
#     total_loss, total_correct, total_tokens = 0, 0, 0
#     for src, tgt in dataloader:
#         src, tgt = src.to(config.device), tgt.to(config.device)
#         tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        
#         optimizer.zero_grad()
#         logits, _ = model(src, tgt_in)
#         loss = criterion(logits.reshape(-1, config.vocab_size), tgt_out.reshape(-1))
#         loss.backward()
#         optimizer.step()
        
#         # 计算按位准确率
#         pred = torch.argmax(logits, dim=-1)
#         mask = (tgt_out != config.vocab['<pad>'])
#         total_correct += (pred == tgt_out).masked_select(mask).sum().item()
#         total_tokens += mask.sum().item()
#         total_loss += loss.item()
    
#     return total_loss/len(dataloader), total_correct/total_tokens if total_tokens else 0

def train_step(model, dataloader, criterion, optimizer, config, epoch):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0

    def get_sampling_prob(epoch, max_epochs):
        if epoch < max_epochs * 0.3:
            return 0.0
        else:
            return min(1.0, (epoch - max_epochs * 0.3) / (max_epochs * 0.7))

    sampling_prob = get_sampling_prob(epoch, config.epochs)

    for src, tgt in dataloader:
        src, tgt = src.to(config.device), tgt.to(config.device)
        batch_size, seq_len = tgt.shape

        tgt_input = torch.full((batch_size, seq_len - 1), config.vocab['<pad>'], dtype=torch.long, device=config.device)
        tgt_input[:, 0] = config.vocab['<s>']

        for t in range(1, seq_len - 1):
            tgt_input[:, t] = tgt[:, t]  # 默认teacher forcing
            use_model_pred = torch.rand(batch_size, device=config.device) < sampling_prob
            idxs = use_model_pred.nonzero(as_tuple=True)[0]

            if len(idxs) > 0 and t > 1:
                with torch.no_grad():
                    logits, _ = model(src[idxs], tgt_input[idxs, :t])
                    pred_token = logits[:, -1, :].argmax(dim=-1)
                tgt_input[idxs, t] = pred_token

        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, config.vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=-1)
        mask = (tgt_out != config.vocab['<pad>'])
        total_correct += (pred == tgt_out).masked_select(mask).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()

    return total_loss / len(dataloader), total_correct / total_tokens if total_tokens else 0


def val_step(model, dataloader, criterion, config):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(config.device), tgt.to(config.device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            
            logits, _ = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, config.vocab_size), tgt_out.reshape(-1))
            
            pred = torch.argmax(logits, dim=-1)
            mask = (tgt_out != config.vocab['<pad>'])
            total_correct += (pred == tgt_out).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()
    
    return total_loss/len(dataloader), total_correct/total_tokens if total_tokens else 0

# 统一的结果解析函数
def parse_sequence(token_ids, config, reverse=True, strip_leading_zeros=True):
    """解析token序列为最终的加法结果字符串"""
    # 截断到 </s>
    if config.vocab['</s>'] in token_ids:
        token_ids = token_ids[:token_ids.index(config.vocab['</s>'])]
    # 转字符并去掉 <pad>
    chars = [config.reverse_vocab[i] for i in token_ids if i != config.vocab['<pad>']]
    # 去掉起始符 <s>
    seq_str = ''.join(chars).replace('<s>', '')
    # 如果需要反转
    if reverse:
        seq_str = seq_str[::-1]
    # 去掉前导零
    if strip_leading_zeros:
        seq_str = seq_str.lstrip('0') or '0'
    return seq_str

def compare_sequences(true_str, pred_str):
    """
    按位对比两个字符串，返回错误位索引列表和对比展示字符串。
    """
    max_len = max(len(true_str), len(pred_str))
    true_str_padded = true_str.ljust(max_len, '_')  # 不够长度用下划线填充
    pred_str_padded = pred_str.ljust(max_len, '_')

    error_positions = []
    compare_str = []
    for i in range(max_len):
        if true_str_padded[i] == pred_str_padded[i]:
            compare_str.append(true_str_padded[i])
        else:
            error_positions.append(i)
            compare_str.append(f"[{pred_str_padded[i]}]")  # 错误位置用[]标记

    return error_positions, ''.join(compare_str)

# # 自回归生成函数
# def autoregressive_generate(model, src, config, max_len=14):
#     model.eval()
#     src = src.to(config.device).unsqueeze(0)  # batch=1
#     generated = [config.vocab['<s>']]
#     with torch.no_grad():
#         for _ in range(max_len):
#             tgt_input = torch.tensor([generated], dtype=torch.long).to(config.device)
#             logits, _ = model(src, tgt_input)
#             next_token = logits[:, -1, :].argmax(dim=-1).item()
#             if next_token == config.vocab['</s>']:
#                 break
#             generated.append(next_token)
#             # 去掉打印，不输出每步输入输出
#     return generated[1:]  # 去掉<s>

def autoregressive_generate(model, src, config, max_len=14):
    model.eval()
    src = src.to(config.device).unsqueeze(0)
    generated = [config.vocab['<s>']]
    with torch.no_grad():
        for step in range(max_len):
            tgt_input = torch.tensor([generated], dtype=torch.long).to(config.device)
            logits, _ = model(src, tgt_input)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == config.vocab['</s>']:
                break
    return [token for token in generated if token not in (config.vocab['</s>'], config.vocab['<s>'])]

# teacher forcing 测试
def test_model_teacher_forcing(model, dataloader, config):
    model.eval()
    samples = []
    total_tokens = 0
    total_correct = 0
    total_full_correct = 0
    total_samples = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(config.device), tgt.to(config.device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            logits, _ = model(src, tgt_in)
            pred = torch.argmax(logits, dim=-1)
            mask = (tgt_out != config.vocab['<pad>'])

            total_correct += (pred == tgt_out).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            for i in range(len(src)):
                true_result = parse_sequence(tgt[i].tolist(), config)
                pred_result = parse_sequence(pred[i].tolist(), config)

                total_full_correct += (true_result == pred_result)
                total_samples += 1

                if len(samples) < 5:
                    src_str = parse_sequence(src[i].tolist(), config, reverse=False, strip_leading_zeros=False)
                    a_rev, b_rev = src_str.split('+')
                    a = a_rev[::-1].lstrip('0') or '0'
                    b = b_rev[::-1].lstrip('0') or '0'
                    samples.append((f"{a}+{b}", true_result, pred_result))

    bit_acc = total_correct / total_tokens if total_tokens else 0
    full_acc = total_full_correct / total_samples if total_samples else 0
    return bit_acc, full_acc, samples


# 自回归测试
def test_model_autoregressive(model, dataloader, config):
    model.eval()
    samples = []
    total_full_correct = 0
    total_samples = 0

    total_tokens = 0
    total_correct = 0

    for src, tgt in dataloader:
        for i in range(len(src)):
            src_i = src[i]
            tgt_i = tgt[i]

            pred_ids = autoregressive_generate(model, src_i, config, max_len=config.max_len)

            true_result = parse_sequence(tgt_i.tolist(), config)
            pred_result = parse_sequence(pred_ids, config)

            # 计算按位准确率
            def clean_ids(ids):
                if config.vocab['</s>'] in ids:
                    ids = ids[:ids.index(config.vocab['</s>'])]
                return [id for id in ids if id != config.vocab['<pad>']]

            tgt_ids_clean = clean_ids(tgt_i.tolist())
            pred_ids_clean = clean_ids(pred_ids)

            compare_len = min(len(tgt_ids_clean), len(pred_ids_clean))
            correct_tokens = sum(1 for j in range(compare_len) if tgt_ids_clean[j] == pred_ids_clean[j])
            total_correct += correct_tokens
            total_tokens += compare_len

            src_str = parse_sequence(src_i.tolist(), config, reverse=False, strip_leading_zeros=False)
            a_rev, b_rev = src_str.split('+')
            a = a_rev[::-1].lstrip('0') or '0'
            b = b_rev[::-1].lstrip('0') or '0'

            total_full_correct += (true_result == pred_result)
            total_samples += 1

            if len(samples) < 5:
                samples.append((f"{a}+{b}", true_result, pred_result))

    bit_acc = total_correct / total_tokens if total_tokens else 0
    full_acc = total_full_correct / total_samples if total_samples else 0
    return bit_acc, full_acc, samples


def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    config = Config()
    print(f"使用设备: {config.device}")
    
    # 加载修复后的数据集
    tokenizer = lambda x: tokenize(x, config.vocab)
    dataset = AddDataset(vocab=config.vocab, tokenizer=tokenizer)
    print(f"数据集大小: {len(dataset)}样本")
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # 数据加载器
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        collate_fn=lambda x: collate_fn(x, src_pad_idx=config.vocab['<pad>'], tgt_pad_idx=config.vocab['<pad>']),
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        collate_fn=lambda x: collate_fn(x, src_pad_idx=config.vocab['<pad>'], tgt_pad_idx=config.vocab['<pad>'])
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        collate_fn=lambda x: collate_fn(x, src_pad_idx=config.vocab['<pad>'], tgt_pad_idx=config.vocab['<pad>'])
    )
    
    # 模型初始化
    model = Transformer(
        src_vocab_size=config.vocab_size,
        tgt_vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_head=config.n_head,
        num_layers=config.num_layers,
        d_hidden=config.d_hidden,
        dropout=config.dropout,
        src_pad_idx=config.vocab['<pad>'],
        tgt_pad_idx=config.vocab['<pad>'],
    ).to(config.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    
    # 训练循环（添加scheduler.step()）
    best_val_loss = float('inf')
    counter = 0
    # for epoch in range(config.epochs):
    #     train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, config)
    #     val_loss, val_acc = val_step(model, val_loader, criterion, config)
    #     scheduler.step()  # 学习率调度生效

    for epoch in range(config.epochs): # scheduled sampling
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, config, epoch)
        val_loss, val_acc = val_step(model, val_loader, criterion, config)
        scheduler.step()
    
        print(f"Epoch {epoch+1} (LR: {optimizer.param_groups[0]['lr']:.6f}):")
        print(f"  训练: 损失={train_loss:.4f}, 按位准确率={train_acc:.4f}")
        print(f"  验证: 损失={val_loss:.4f}, 按位准确率={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.save_path)
            counter = 0
        else:
            counter += 1
            if counter >= config.patience:
                print("早停！")
                break
    

    # 测试
    print("\n=== 非自回归测试（teacher forcing） ===")
    bit_acc, full_acc_tf, samples_tf = test_model_teacher_forcing(model, test_loader, config)
    print(f"按位准确率: {bit_acc:.4f}")
    print(f"全序列准确率: {full_acc_tf:.4f}")
    print("预测示例:")
    for src, true, pred in samples_tf:
        error_pos, compare_str = compare_sequences(true, pred)
        print(f"输入: {src} | 真实: {true} | 预测: {pred}")
        if error_pos:
            print(f"  位置错误（teacher forcing）: {error_pos}, 对比: {compare_str}")
        else:
            print("  完全正确")

    print("\n=== 自回归生成测试 ===")
    bit_acc_ar, full_acc_ar, samples_ar = test_model_autoregressive(model, test_loader, config)
    print(f"按位准确率: {bit_acc_ar:.4f}")
    print(f"全序列准确率: {full_acc_ar:.4f}")
    print("预测示例:")
    for src, true, pred in samples_ar:
        error_pos, compare_str = compare_sequences(true, pred)
        print(f"输入: {src} | 真实: {true} | 预测: {pred}")
        if error_pos:
            print(f"  位置错误（自回归）: {error_pos}, 对比: {compare_str}")
        else:
            print("  完全正确")






if __name__ == "__main__":
    main()