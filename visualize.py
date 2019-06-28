import matplotlib.pyplot as plt
import numpy as np

pruning_rate = 0.2


def plot_test_accuracy(test_accuracy_history):
    plt.subplot(2, 1, 1)
    axis = plt.plot(test_accuracy_history)

    # Retrieve indexes at which iterative pruning/retraining was performed
    pruning_iterations = np.arange(len(test_accuracy_history))
    # Finding percentage of weights remaining from the iteration

    weights_sparsity_history = np.around(100 * (1 - pruning_rate) ** (pruning_iterations + 1), decimals=1)

    pretty_pruning_iterations = []
    pretty_weights_sparsity_history = []
    last_weight = 100

    for cur_idx in np.nditer(pruning_iterations):
        cur_weight = weights_sparsity_history[cur_idx]
        if cur_weight <= last_weight/2:  # If the remaining weights is at most half as last one, add for graphing
            pretty_pruning_iterations.append(cur_idx)
            pretty_weights_sparsity_history.append(cur_weight)
            last_weight = cur_weight

    plt.xticks(pretty_pruning_iterations, pretty_weights_sparsity_history)
    plt.grid()
    plt.xlabel('Percentage of weights remaining')
    plt.ylabel('Test accuracy')
    plt.show()
    plt.savefig('test_results_iter_pruning.png')


#accuracy_hist = [83.6, 86.4, 87.8, 87.3, 84.4,   83.0, 83.5, 82.1, 82.9, 79.1, 79.1, 78.0, 78.1, 78.0, 78.1, 78.0, 79.1, 78.0]
#plot_test_accuracy(accuracy_hist)