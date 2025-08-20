import csv
import random
from feature_batch import Random_embedding, Glove_embedding
import torch
from comparison_plot_batch import NN_embedding_plot

# 数据读入
with open('../external_resources/new_train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp_train = list(tsvreader)

with open('../external_resources/new_test.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp_test = list(tsvreader)

with open('../glove/glove.6B.300d.txt', encoding='utf-8') as f:
    lines = f.readlines()

d_model = 300

# 用Glove创建词典
trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()                                                                  # 将lines分成n行，每行是一个词和它的向量
    trained_dict[line[0].upper()] = [float(line[j]) for j in range(1, d_model+1)]   # line[0]是词，line[1:]是向量, [1,51)共50维

# 初始化
iter_times = 50
alpha = 1e-3
# 程序开始
train_data = temp_train[1:]                                                                              # 去掉表头
test_data = temp_test[1:]
batch_size = 500
rnn_layer = 2
drop_out = 0.1
patience = 5

# 随机初始化
random.seed(2025)
random_embedding = Random_embedding(train_data=train_data, test_data=test_data)
random_embedding.get_words()
random_embedding.get_id()

#Glove初始化
glove_embedding = Glove_embedding(train_data=train_data, test_data=test_data, trained_dict=trained_dict, d_model=d_model)
glove_embedding.get_words()
glove_embedding.get_id()

NN_embedding_plot(random_embedding, glove_embedding, alpha, batch_size, iter_times, d_model, rnn_layer, drop_out, patience)          # 绘制图像