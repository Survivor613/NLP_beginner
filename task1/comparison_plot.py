import matplotlib.pyplot as plt
from mysoftmax_regression import Softmax

def alpha_gradient_plot(bag,gram,total_times,mini_size):
    """Plot categorizaiton verses different parameters"""
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    #Bag of Words
    
    #Shuffle 随机梯度下降
    bag_shuffle_train = list()
    bag_shuffle_val = list()
    bag_shuffle_test = list()
    for alpha in alphas:
        soft11 = Softmax(len(bag.train), 5, bag.len)
        soft11.regression(bag.train_matrix, bag.train_y, alpha, total_times, "shuffle")
        train_result, val_result, test_result = soft11.correct_rate(bag.train_matrix, bag.train_y, bag.val_matrix, bag.val_y, bag.test_matrix, bag.test_y)
        bag_shuffle_train.append(train_result)
        bag_shuffle_val.append(val_result)
        bag_shuffle_test.append(test_result)
    print('\n')
        
    # Batch 批量梯度下降
    bag_batch_train = []
    bag_batch_val = []
    bag_batch_test = []
    for alpha in alphas:
        soft12 = Softmax(len(bag.train), 5, bag.len)
        soft12.regression(bag.train_matrix, bag.train_y, alpha, total_times, "batch")
        train_result, val_result, test_result = soft12.correct_rate(
            bag.train_matrix, bag.train_y,
            bag.val_matrix, bag.val_y,
            bag.test_matrix, bag.test_y
        )
        bag_batch_train.append(train_result)
        bag_batch_val.append(val_result)
        bag_batch_test.append(test_result)
    print('\n')

    # Mini-batch 小批量梯度下降
    bag_mini_batch_train = []
    bag_mini_batch_val = []
    bag_mini_batch_test = []
    for alpha in alphas:
        soft13 = Softmax(len(bag.train), 5, bag.len)
        soft13.regression(bag.train_matrix, bag.train_y, alpha, total_times, "mini", mini_size)
        train_result, val_result, test_result = soft13.correct_rate(
            bag.train_matrix, bag.train_y,
            bag.val_matrix, bag.val_y,
            bag.test_matrix, bag.test_y
        )
        bag_mini_batch_train.append(train_result)
        bag_mini_batch_val.append(val_result)
        bag_mini_batch_test.append(test_result)
    print('\n')

    
    # 绘图
    plt.subplot(2, 3, 1) # create a 2x2 subplot and define it as the 1st subplot
    plt.semilogx(alphas, bag_shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, bag_shuffle_train, 'ro-')  # add some point to the chart while keeping the label simple.
    plt.semilogx(alphas, bag_batch_train, 'g--', label='batch')
    plt.semilogx(alphas, bag_batch_train, 'g+-')
    plt.semilogx(alphas, bag_mini_batch_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, bag_mini_batch_train, 'b^-')
    plt.legend() # show legend(e.g. red:shuffle, green:batch, blue:mini-batch)
    plt.title("Bag of Words -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    
    plt.subplot(2, 3, 2)
    plt.semilogx(alphas, bag_shuffle_val, 'r--', label='shuffle')
    plt.semilogx(alphas, bag_shuffle_val, 'ro-')
    plt.semilogx(alphas, bag_batch_val, 'g--', label='batch')
    plt.semilogx(alphas, bag_batch_val, 'go--')
    plt.semilogx(alphas, bag_mini_batch_val, 'b--', label='mini-batch')
    plt.semilogx(alphas, bag_mini_batch_val, 'b^-')
    plt.legend()
    plt.title("Bag of Words -- Val Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)

    plt.subplot(2, 3, 3)
    plt.semilogx(alphas, bag_shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, bag_shuffle_test, 'ro-')
    plt.semilogx(alphas, bag_batch_test, 'g--', label='batch')
    plt.semilogx(alphas, bag_batch_test, 'go--')
    plt.semilogx(alphas, bag_mini_batch_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, bag_mini_batch_test, 'b^-')
    plt.legend()
    plt.title("Bag of Words -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    
    # N-gram

    # Shuffle 随机梯度下降
    gram_shuffle_train = []
    gram_shuffle_val = []
    gram_shuffle_test = []
    for alpha in alphas:
        soft21 = Softmax(len(gram.train), 5, gram.len)
        soft21.regression(gram.train_matrix, gram.train_y, alpha, total_times, "shuffle")
        train_result, val_result, test_result = soft21.correct_rate(
            gram.train_matrix, gram.train_y,
            gram.val_matrix, gram.val_y,
            gram.test_matrix, gram.test_y
        )
        gram_shuffle_train.append(train_result)
        gram_shuffle_val.append(val_result)
        gram_shuffle_test.append(test_result)
    print('\n')

    # Batch 批量梯度下降
    gram_batch_train = []
    gram_batch_val = []
    gram_batch_test = []
    for alpha in alphas:
        soft22 = Softmax(len(gram.train), 5, gram.len)
        soft22.regression(gram.train_matrix, gram.train_y, alpha, total_times, "batch")
        train_result, val_result, test_result = soft22.correct_rate(
            gram.train_matrix, gram.train_y,
            gram.val_matrix, gram.val_y,
            gram.test_matrix, gram.test_y
        )
        gram_batch_train.append(train_result)
        gram_batch_val.append(val_result)
        gram_batch_test.append(test_result)
    print('\n')

    # Mini-batch 小批量梯度下降
    gram_mini_batch_train = []
    gram_mini_batch_val = []
    gram_mini_batch_test = []
    for alpha in alphas:
        soft23 = Softmax(len(gram.train), 5, gram.len)
        soft23.regression(gram.train_matrix, gram.train_y, alpha, total_times, "mini", mini_size)
        train_result, val_result, test_result = soft23.correct_rate(
            gram.train_matrix, gram.train_y,
            gram.val_matrix, gram.val_y,
            gram.test_matrix, gram.test_y
        )
        gram_mini_batch_train.append(train_result)
        gram_mini_batch_val.append(val_result)
        gram_mini_batch_test.append(test_result)
    print('\n')


    # 绘图
    plt.subplot(2, 3, 4) # create a 2x2 subplot and define it as the 1st subplot
    plt.semilogx(alphas, gram_shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, gram_shuffle_train, 'ro-')  # add some point to the chart while keeping the label simple.
    plt.semilogx(alphas, gram_batch_train, 'g--', label='batch')
    plt.semilogx(alphas, gram_batch_train, 'g+-')
    plt.semilogx(alphas, gram_mini_batch_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, gram_mini_batch_train, 'b^-')
    plt.legend() # show legend(e.g. red:shuffle, green:batch, blue:mini-batch)
    plt.title(f"N-gram({gram.dimension}-gram) -- Train Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    
    plt.subplot(2, 3, 5)
    plt.semilogx(alphas, gram_shuffle_val, 'r--', label='shuffle')
    plt.semilogx(alphas, gram_shuffle_val, 'ro-')
    plt.semilogx(alphas, gram_batch_val, 'g--', label='batch')
    plt.semilogx(alphas, gram_batch_val, 'go--')
    plt.semilogx(alphas, gram_mini_batch_val, 'b--', label='mini-batch')
    plt.semilogx(alphas, gram_mini_batch_val, 'b^-')
    plt.legend()
    plt.title(f"N-gram({gram.dimension}-gram) -- Val Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)

    plt.subplot(2, 3, 6)
    plt.semilogx(alphas, gram_shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, gram_shuffle_test, 'ro-')
    plt.semilogx(alphas, gram_batch_test, 'g--', label='batch')
    plt.semilogx(alphas, gram_batch_test, 'go--')
    plt.semilogx(alphas, gram_mini_batch_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, gram_mini_batch_test, 'b^-')
    plt.legend()
    plt.title(f"N-gram({gram.dimension}-gram) -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    


    
    # 展示
    plt.show()