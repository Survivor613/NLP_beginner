import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from Neural_network_batch import MY_CNN, MY_RNN
from feature_batch import get_batch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有GPU可用

class EarlyStopping:
    """早停工具类"""
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class WarmupInverseSqrtScheduler:
    """
    Warmup + Inverse sqrt decay
    """
    def __init__(self, optimizer, d_model, warmup_steps=2000, scale=1.0):
        """
        scale: 用于调整最终学习率峰值
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.scale = scale

    def step(self):
        self.step_num += 1
        lr = self.scale * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



def NN_embdding(model, train, val, test, learning_rate, iter_times, d_model, patience=5):
    # 定义优化器（求参数）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 定义学习率调度器
    scheduler = WarmupInverseSqrtScheduler(optimizer, d_model=d_model, warmup_steps=1000, scale=2.0)

    # 定义损失函数
    loss_fun = F.cross_entropy
    # 记录训练集和测试集的loss和accuracy
    train_loss_record = list()
    val_loss_record = list()
    test_loss_record = list()
    long_loss_record = list()

    train_record = list() # accuracy
    val_record = list()
    test_record = list()
    long_record = list()

    early_stopping = EarlyStopping(patience=patience, path="best_model.pt")

    # 训练阶段
    for iteration in range(iter_times): # 1个iteration即为1个epoch
        model.train()  # 设置模型为训练模式
        for i, batch in enumerate(train):
            x, y = batch
            x = x.to(device)  # 将数据移动到GPU或CPU
            y = y.to(device)
            pred = model(x)  # 前向传播
            optimizer.zero_grad()  # 清除梯度
            loss = loss_fun(pred, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        scheduler.step() # 更新学习率
        
        model.eval()  # 设置模型为评估模式
        # 本轮正确率记录
        train_acc = list() # 每个batch的平均正确率
        val_acc = list()
        long_acc = list()
        test_acc = list()
        length = 20
        # 本轮损失值记录
        train_loss = 0 # 每个epoch中每个batch的loss之和
        val_loss = 0
        test_loss = 0
        long_loss = 0
        for i, batch in enumerate(train):
            x, y = batch
            x = x.to(device)  # 将数据移动到GPU或CPU
            y = y.to(device)
            pred = model(x)
            loss = loss_fun(pred, y)
            train_loss += loss.item() # item()将张量类型(torch.tensor[...])转换为数值类型
            # 取pred的最大值，-1代表在最后一个维度（类别维度）求最大值(pred此处是一个矩阵)，_代表忽略最大值，只保留最大值所在的索引
            _, y_pre = torch.max(pred, -1)
            #计算本batch的准确率
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            train_acc.append(acc)
        
        for i, batch in enumerate(val):
            x, y = batch
            x = x.to(device)  # 将数据移动到GPU或CPU
            y = y.to(device)
            pred = model(x)
            loss = loss_fun(pred, y)
            val_loss += loss.item() # 调用item()以后就会变成Python数值,而Python数值只能存储于CPU中
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            val_acc.append(acc)

        for i, batch in enumerate(test):
            x, y = batch
            x = x.to(device)  # 将数据移动到GPU或CPU
            y = y.to(device)
            pred = model(x)
            loss = loss_fun(pred, y)
            test_loss += loss.item() # 调用item()以后就会变成Python数值,而Python数值只能存储于CPU中
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            test_acc.append(acc)

            # 补充进行针对长句子的侦测
            if len(x[0]) > length:
                long_loss += loss.item()
                long_acc.append(acc)

        trains_loss = train_loss / len(train_acc) # 除以epoch数量，对epoch求平均
        vals_loss = val_loss / len(val_acc)
        tests_loss = test_loss / len(test_acc)
        longs_loss = long_loss / len(long_acc)
        train_loss_record.append(trains_loss)
        val_loss_record.append(vals_loss)
        test_loss_record.append(tests_loss)
        long_loss_record.append(longs_loss)

        trains_acc = sum(train_acc) / len(train_acc)  # 除以epoch数量，求每个epoch的平均准确率
        vals_acc = sum(val_acc) / len(val_acc)
        tests_acc = sum(test_acc) / len(test_acc)
        longs_acc = sum(long_acc) / len(long_acc)
        train_record.append(trains_acc)
        val_record.append(vals_acc)
        long_record.append(longs_acc)
        test_record.append(tests_acc)

        # 输出当前epoch的训练集和测试集loss和accuracy
        print(f"---------- Iteration {iteration + 1} ----------")
        print(f"Train Loss: {trains_loss:.4f}")
        print(f"Val Loss: {vals_loss:.4f}")
        print(f"Test Loss: {tests_loss:.4f}")
        print(f"Long Loss: {longs_loss:.4f}")
        print(f"Train Accuracy: {trains_acc:.4f}")
        print(f"Val Accuracy: {vals_acc:.4f}")
        print(f"Test Accuracy: {tests_acc:.4f}")
        print(f"Long Accuracy: {longs_acc:.4f}")

        # ====== 调用早停器 ======
        early_stopping(vals_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    model.load_state_dict(torch.load("best_model.pt"))

    print("Training complete.")
    
    return train_loss_record, val_loss_record, test_loss_record, long_loss_record, train_record, val_record, test_record, long_record



def NN_embedding_plot(random_embedding, glove_embedding, learning_rate, batch_size, iter_times, d_model, rnn_layer, drop_out, patience=5):
    # 获取训练集和测试集的batch
    train_random = get_batch(random_embedding.train_matrix, random_embedding.train_y, batch_size)
    train_glove = get_batch(glove_embedding.train_matrix, glove_embedding.train_y, batch_size)
    val_random = get_batch(random_embedding.val_matrix, random_embedding.val_y, batch_size)
    val_glove = get_batch(glove_embedding.val_matrix, glove_embedding.val_y, batch_size)
    test_random = get_batch(random_embedding.test_matrix, random_embedding.test_y, batch_size)
    test_glove = get_batch(glove_embedding.test_matrix, glove_embedding.test_y, batch_size)

    # 模型建立
    torch.manual_seed(2025)
    print(random_embedding.longest, type(random_embedding.longest))
    random_cnn = MY_CNN(d_model, random_embedding.len_words, random_embedding.longest, drop_out=drop_out)
    glove_cnn = MY_CNN(d_model, glove_embedding.len_words, glove_embedding.longest, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float), drop_out=drop_out)
    random_rnn = MY_RNN(d_model, d_model, random_embedding.len_words, layer=rnn_layer, drop_out=drop_out)
    glove_rnn = MY_RNN(d_model, d_model, glove_embedding.len_words, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float), layer=rnn_layer, drop_out=drop_out)

    # cnn + random
    torch.manual_seed(2025)
    trl_ran_cnn, val_ran_cnn, tel_ran_cnn, lol_ran_cnn, tra_ran_cnn, vaa_ran_cnn, tea_ran_cnn, loa_ran_cnn = NN_embdding(random_cnn, train_random, val_random, test_random, learning_rate, iter_times, d_model, patience)
    # 注: trl_ran_cnn, val_ran_cnn, lol_ran_cnn, tra_ran_cnn, vaa_ran_cnn, loa_ran_cnn 分别为训练集loss，测试集loss，长句loss，训练集accuracy，测试集accuracy，长句accuracy

    # cnn + glove
    torch.manual_seed(2025)
    trl_glo_cnn, val_glo_cnn, tel_glo_cnn, lol_glo_cnn, tra_glo_cnn, vaa_glo_cnn, tea_glo_cnn, loa_glo_cnn = NN_embdding(glove_cnn, train_glove, val_glove, test_glove, learning_rate, iter_times, d_model, patience)

    # rnn + random
    torch.manual_seed(2025)
    trl_ran_rnn, val_ran_rnn, tel_ran_rnn, lol_ran_rnn, tra_ran_rnn, vaa_ran_rnn, tea_ran_rnn, loa_ran_rnn = NN_embdding(random_rnn, train_random, val_random, test_random, learning_rate, iter_times, d_model, patience)

    # rnn + glove
    torch.manual_seed(2025)
    trl_glo_rnn, val_glo_rnn, tel_glo_rnn, lol_glo_rnn, tra_glo_rnn, vaa_glo_rnn, tea_glo_rnn, loa_glo_rnn = NN_embdding(glove_rnn, train_glove, val_glove, test_glove, learning_rate, iter_times, d_model, patience)






    # 画图部分
    def prepare_for_plot(data):
        """预处理函数,将GPU上的张量/数值列表转换为CPU上的numpy数组"""
        processed = []
        for item in data:
            # 如果是张量列表
            if isinstance(item, torch.Tensor):
                processed_item = item.cpu().detach().numpy() # cpu()将数据从GPU上转换到CPU上, detach()切断与计算图的链接
                processed.append(processed_item)
            # 如果是数值列表
            else:
                processed.append(item)
        # 将列表转为 NumPy 数组（方便绘图）
        return np.array(processed)
    
    # 预处理,将GPU上的张量转换为CPU上的numpy数组
    trl_ran_cnn = prepare_for_plot(trl_ran_cnn)
    val_ran_cnn = prepare_for_plot(val_ran_cnn)
    tel_ran_cnn = prepare_for_plot(tel_ran_cnn)
    lol_ran_cnn = prepare_for_plot(lol_ran_cnn)
    tra_ran_cnn = prepare_for_plot(tra_ran_cnn)
    vaa_ran_cnn = prepare_for_plot(vaa_ran_cnn)
    tea_ran_cnn = prepare_for_plot(tea_ran_cnn)
    loa_ran_cnn = prepare_for_plot(loa_ran_cnn)

    trl_glo_cnn = prepare_for_plot(trl_glo_cnn)
    val_glo_cnn = prepare_for_plot(val_glo_cnn)
    tel_glo_cnn = prepare_for_plot(tel_glo_cnn)
    lol_glo_cnn = prepare_for_plot(lol_glo_cnn)
    tra_glo_cnn = prepare_for_plot(tra_glo_cnn)
    vaa_glo_cnn = prepare_for_plot(vaa_glo_cnn)
    tea_glo_cnn = prepare_for_plot(tea_glo_cnn)
    loa_glo_cnn = prepare_for_plot(loa_glo_cnn)

    trl_ran_rnn = prepare_for_plot(trl_ran_rnn)
    val_ran_rnn = prepare_for_plot(val_ran_rnn)
    tel_ran_rnn = prepare_for_plot(tel_ran_rnn)
    lol_ran_rnn = prepare_for_plot(lol_ran_rnn)
    tra_ran_rnn = prepare_for_plot(tra_ran_rnn)
    vaa_ran_rnn = prepare_for_plot(vaa_ran_rnn)
    tea_ran_rnn = prepare_for_plot(tea_ran_rnn)
    loa_ran_rnn = prepare_for_plot(loa_ran_rnn)

    trl_glo_rnn = prepare_for_plot(trl_glo_rnn)
    val_glo_rnn = prepare_for_plot(val_glo_rnn)
    tel_glo_rnn = prepare_for_plot(tel_glo_rnn)
    lol_glo_rnn = prepare_for_plot(lol_glo_rnn)
    tra_glo_rnn = prepare_for_plot(tra_glo_rnn)
    vaa_glo_rnn = prepare_for_plot(vaa_glo_rnn)
    tea_glo_rnn = prepare_for_plot(tea_glo_rnn)
    loa_glo_rnn = prepare_for_plot(loa_glo_rnn)

    # 正式绘制图像
    def plot_curve(y, style, label): # 辅助函数
        plt.plot(range(1, len(y) + 1), y, style, label=label)

    # --- 主图 (6个子图) ---
    plt.subplot(2, 3, 1)
    plot_curve(trl_ran_rnn, 'r--', 'RNN+random')
    plot_curve(trl_ran_cnn, 'g--', 'CNN+random')
    plot_curve(trl_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(trl_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.subplot(2, 3, 2)
    plot_curve(val_ran_rnn, 'r--', 'RNN+random')
    plot_curve(val_ran_cnn, 'g--', 'CNN+random')
    plot_curve(val_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(val_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Val Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.subplot(2, 3, 3)
    plot_curve(tel_ran_rnn, 'r--', 'RNN+random')
    plot_curve(tel_ran_cnn, 'g--', 'CNN+random')
    plot_curve(tel_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(tel_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.subplot(2, 3, 4)
    plot_curve(tra_ran_rnn, 'r--', 'RNN+random')
    plot_curve(tra_ran_cnn, 'g--', 'CNN+random')
    plot_curve(tra_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(tra_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Train Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.subplot(2, 3, 5)
    plot_curve(vaa_ran_rnn, 'r--', 'RNN+random')
    plot_curve(vaa_ran_cnn, 'g--', 'CNN+random')
    plot_curve(vaa_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(vaa_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Val Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.subplot(2, 3, 6)
    plot_curve(tea_ran_rnn, 'r--', 'RNN+random')
    plot_curve(tea_ran_cnn, 'g--', 'CNN+random')
    plot_curve(tea_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(tea_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Test Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig('main_plot.jpg')
    plt.show()

    # --- 子图 (长句实验 2个子图) ---
    plt.subplot(2, 1, 1)
    plot_curve(loa_ran_rnn, 'r--', 'RNN+random')
    plot_curve(loa_ran_cnn, 'g--', 'CNN+random')
    plot_curve(loa_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(loa_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Long Sentence Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.subplot(2, 1, 2)
    plot_curve(lol_ran_rnn, 'r--', 'RNN+random')
    plot_curve(lol_ran_cnn, 'g--', 'CNN+random')
    plot_curve(lol_glo_rnn, 'b--', 'RNN+glove')
    plot_curve(lol_glo_cnn, 'y--', 'CNN+glove')
    plt.legend(fontsize=10)
    plt.title("Long Sentence Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig('sub_plot.jpg')
    plt.show()
