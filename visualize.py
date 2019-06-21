import matplotlib.pyplot as plt


def plot_test_accuracy(sparsity_weights_history, test_accuracy_history):
    plt.subplot(2, 1, 1)
    plt.plot(sparsity_weights_history, test_accuracy_history)
    #plt.xticks(sparsity_weights_history, sparsity_weights_history)
    plt.xlabel('Percentage of weights remaining')
    plt.ylabel('Test accuracy')
    plt.show()
    plt.savefig('test_results_iter_pruning.png')


plot_test_accuracy([100,50,10,2], [95, 95, 94, 93])