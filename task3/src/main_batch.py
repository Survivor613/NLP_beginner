from feature_batch import Random_embedding, Glove_embedding
import random
from comparison_plot_batch import NN_plot

with open('../external_resources/snli_1.0/snli_1.0_train.txt', 'r') as f:
    temp = f.readlines()

with open('../external_resources/glove/glove.6B.50d.txt','rb') as f:
    lines = f.readlines()

# 构建embedding字典

trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()  # 取出glove中的第i行,如 "the 0.418 0.24968 ...",返回一个列表，如 ['the', '0.418', '0.24968', ...]
    trained_dict[line[0].upper()] = [float(line[j]) for j in range(1, 51)]

data = temp[1:]
learning_rate = 0.001
len_feature = 50
len_hidden = 50
iter_times = 50
batch_size = 1000

# random embedding
random.seed(2025)
random_embedding = Random_embedding(data=data) # 将一句话变为ID序列
random_embedding.get_words()
random_embedding.get_id()

# glove embedding
random.seed(2025)
glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()

NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times)