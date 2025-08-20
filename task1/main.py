import numpy
import csv
import random
from feature import Bag, Gram
from comparison_plot import alpha_gradient_plot

with open('new_train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp_train = list(tsvreader)  # turn temp into a list

with open('new_test.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp_test = list(tsvreader)  # turn temp into a list
    
train_data = temp_train[1:]
test_data = temp_test[1:]
max_item = 2000
random.seed(2023)
numpy.random.seed(2023)

bag = Bag(train_data, test_data, max_item)
bag.get_words()
bag.get_matrix()

gram = Gram(train_data, test_data, dimension=2,max_item=max_item)
gram.get_words()
gram.get_matrix()

alpha_gradient_plot(bag, gram, 10000, 10) # 10000:total_times, 10:mini_size