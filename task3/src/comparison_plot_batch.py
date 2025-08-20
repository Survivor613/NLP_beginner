import matplotlib.pyplot
import torch
import torch.nn.functional as F
from feature_batch import get_batch
from torch import optim
from Neural_Network_batch import ESIM
import random
import numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NN_embdding(model, train, test,learning_rate, iter_times):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = F.cross_entropy # 把交叉熵函数本身赋给loss_fun
    train_loss_record = list()
    test_loss_record = list()
    train_record = list()
    test_record = list()
    
    model.to(device)

    for iteration in range(iter_times):
        torch.cuda.empty_cache()
        model.train() # 设置为训练模式
        for batch in train:
            torch.cuda.empty_cache()
            x1, x2, y = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            pred = model(x1, x2) # 得到预测向量(未Softmax)
            optimizer.zero_grad()
            loss = loss_fun(pred, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval() # 设置为评估模式
            train_acc = list()
            test_acc = list()
            train_loss = 0
            test_loss = 0
            for batch in train:
                x1, x2, y = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                pred = model(x1, x2)
                loss = loss_fun(pred, y)
                train_loss += loss.item()
                _ , y_pre = torch.max(pred, -1) # pred维度为[sequence_length, type_num],-1代表在type_num这一行中取最大值, max返回(input, dim),我们只关心索引
                acc = torch.mean((torch.tensor(y_pre.cpu() == y.cpu(), dtype=torch.float)))
                train_acc.append(acc)
            for batch in test:
                x1, x2, y = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                pred = model(x1, x2)
                loss = loss_fun(pred, y)
                test_loss += loss.item()
                _ , y_pre = torch.max(pred, -1) # pred维度为[sequence_length, type_num],-1代表在type_num这一行中取最大值, max返回(input, dim),我们只关心索引
                acc = torch.mean((torch.tensor(y_pre.cpu() == y.cpu(), dtype=torch.float)))
                test_acc.append(acc)
        
        trains_loss = train_loss / len(train_acc)
        tests_loss = test_loss / len(test_acc)
        trains_acc = (sum(train_acc) / len(train_acc)).item()
        tests_acc = (sum(test_acc) / len(test_acc)).item()

        train_loss_record.append(trains_loss) # 平均每个batch的loss
        test_loss_record.append(tests_loss)
        train_record.append(trains_acc)
        test_record.append(tests_acc)

        print("---------- Iteration", iteration + 1, "----------")
        print("Train loss:", trains_loss)
        print("Test loss:", tests_loss)
        print("Train accuracy:", trains_acc)
        print("Test accuracy:", tests_acc)

    return train_loss_record, test_loss_record, train_record, test_record


def NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times):
    train_random = get_batch(random_embedding.train_s1_matrix, random_embedding.train_s2_matrix, random_embedding.train_y, batch_size)
    test_random = get_batch(random_embedding.test_s1_matrix, random_embedding.test_s2_matrix, random_embedding.test_y, batch_size)
    train_glove = get_batch(glove_embedding.train_s1_matrix, glove_embedding.train_s2_matrix, glove_embedding.train_y, batch_size)
    test_glove = get_batch(glove_embedding.test_s1_matrix, glove_embedding.test_s2_matrix, glove_embedding.test_y, batch_size)

    random.seed(2025)
    numpy.random.seed(2025)
    torch.manual_seed(2025)
    random_model = ESIM(len_feature, len_hidden, random_embedding.len_words, longest=random_embedding.longest) # longest疑似多余
    
    random.seed(2025)
    numpy.random.seed(2025)
    torch.manual_seed(2025)
    glove_model = ESIM(len_feature, len_hidden, glove_embedding.len_words, longest=glove_embedding.longest, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))

    random.seed(2025)
    numpy.random.seed(2025)
    torch.manual_seed(2025)
    trl_ran, tel_ran, tra_ran, tea_ran = NN_embdding(random_model, train_random, test_random, learning_rate, iter_times)
    
    random.seed(2025)
    numpy.random.seed(2025)
    torch.manual_seed(2025)
    trl_glo, tel_glo, tra_glo, tea_glo = NN_embdding(glove_model, train_glove, test_glove, learning_rate, iter_times)



#############################################################################################################################################################################################
# 可视化
    x = list(range(1, iter_times + 1))
    matplotlib.pyplot.subplot(2, 2, 1)
    matplotlib.pyplot.plot(x, trl_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, trl_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 2)
    matplotlib.pyplot.plot(x, tel_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tel_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 3)
    matplotlib.pyplot.plot(x, tra_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tra_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 2, 4)
    matplotlib.pyplot.plot(x, tea_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tea_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward=True)
    matplotlib.pyplot.savefig('main_plot.jpg')
    matplotlib.pyplot.show()