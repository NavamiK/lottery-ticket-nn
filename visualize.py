import matplotlib.pyplot as plt
import numpy as np

pruning_rate = 0.2

def plot_test_accuracy_coarse(test_accuracy_history1, test_accuracy_history_subset):

    # Retrieve indexes at which iterative pruning/retraining was performed
    pruning_iterations = np.arange(max(len(test_accuracy_history1),len(test_accuracy_history1)))

    # Plotting in bigger intervals
    weights_sparsity_history = np.around(100 * (1 - pruning_rate) ** (pruning_iterations + 1), decimals=1)
    plt.plot(weights_sparsity_history, test_accuracy_history1, label="Using 100% of training set to generate winning ticket - base expt")
    plt.gca().invert_xaxis()

    # If second history is not empty, plot it too
    if test_accuracy_history_subset:
        for subset_size, test_accuracy in test_accuracy_history_subset.items():
            plt.plot(weights_sparsity_history, test_accuracy, label="Using " + subset_size +
                                                                    "% of training set to generate winning ticket")
        plt.legend()

    plt.title("Test accuracy for winning tickets - summary ")
    plt.grid()
    plt.xlabel('Percentage of weights remaining')
    plt.ylabel('Test accuracy')
    plt.show()
    plt.savefig('test_results_iter_pruning_coarse.png')


def plot_test_accuracy_fine(test_accuracy_history1, test_accuracy_history2):
    #plt.subplot(2, 1, 1)

    # Retrieve indexes at which iterative pruning/retraining was performed
    pruning_iterations = np.arange(max(len(test_accuracy_history1),len(test_accuracy_history1)))
    pretty_pruning_iterations, pretty_weights_sparsity_history = get_x_axis_entries(pruning_iterations)

    plt.plot(test_accuracy_history1, label="Using 100% of training set to generate winning ticket - base expt")

    # If second history is not empty, plot it too
    if test_accuracy_history2:
        plt.plot(test_accuracy_history2, label="Using 10% of training set to generate winning ticket")
        plt.legend()

    # Plotting in smaller intervals
    plt.title("Test accuracy for winning tickets - detailed")
    plt.xticks(pretty_pruning_iterations, pretty_weights_sparsity_history)
    plt.grid()
    plt.xlabel('Percentage of weights remaining')
    plt.ylabel('Test accuracy')
    plt.show()
    plt.savefig('test_results_iter_pruning_fine.png')


def get_x_axis_entries(pruning_iterations):

    # Finding percentage of weights remaining from the iteration
    weights_sparsity_history = np.around(100 * (1 - pruning_rate) ** (pruning_iterations + 1), decimals=1)

    pretty_pruning_iterations = []
    pretty_weights_sparsity_history = []
    last_weight = 100

    for cur_idx in np.nditer(pruning_iterations):
        cur_weight = weights_sparsity_history[cur_idx]
        if cur_weight <= last_weight / 2:  # If the remaining weights is at most half as last one, add for graphing
            pretty_pruning_iterations.append(cur_idx)
            pretty_weights_sparsity_history.append(cur_weight)
            last_weight = cur_weight

    return pretty_pruning_iterations, pretty_weights_sparsity_history


#accuracy_hist = [83.6, 86.4, 87.8, 87.3, 84.4,   83.0, 83.5, 82.1, 82.9, 79.1, 79.1, 78.0, 78.1, 78.0, 78.1, 78.0, 79.1, 78.0]
#accuracy_hist1 = [97.8, 97.8, 98.1, 97.5, 97.4, 98.0, 97.8, 97.7, 97.0, 97.2, 97.2, 97.9, 97.3, 97.5, 96.6, 96.8, 96.2, 96.3, 95.6, 94.7, 94.3, 93.1, 93.4, 89.7, 89.0, 87.2, 86.5]
# Result for experiment - see experiment1
#accuracy_hist2 = [96.9, 97.9, 97.6, 97.4, 98.1, 97.4, 97.8, 97.4, 97.9, 97.3, 97.3, 97.1, 96.7, 96.3, 96.1, 95.7, 94.7, 94.3, 93.7, 91.7, 90.4, 85.7, 83.4, 79.5, 72.2, 65.3, 56.1]
accuracy_hist = [97.8, 97.8, 98.1, 97.5, 97.4, 98.0, 97.8, 97.7, 97.0, 97.2, 97.2, 97.9, 97.3, 97.5, 96.6, 96.8, 96.2, 96.3, 95.6, 94.7, 94.3, 93.1, 93.4, 89.7, 89.0, 87.2, 86.5]
accuracy_hist_subset_test = {"5": [99, 99, 99],
                        "10": [98, 98, 98],
                        "20": [97, 97, 97],
                        "40": [96, 96, 96],
                        "80": [95, 95, 95]
                        }
accuracy_hist_subset = {
                        "10": [96.9, 97.9, 97.6, 97.4, 98.1, 97.4, 97.8, 97.4, 97.9, 97.3, 97.3, 97.1, 96.7, 96.3, 96.1, 95.7, 94.7, 94.3, 93.7, 91.7, 90.4, 85.7, 83.4, 79.5, 72.2, 65.3, 56.1],
                        "20": [97.5, 98.3, 97.7, 97.8, 97.9, 97.6, 97.3, 98.0, 97.0, 97.2, 97.7, 97.0, 96.6, 97.0, 96.3, 95.0, 95.4, 94.8, 94.5, 93.9, 93.7, 91.9, 89.0, 86.2, 74.6, 56.3, 47.1],
                        "40": [97.7, 97.9, 98.1, 98.0, 98.0, 97.6, 97.5, 97.7, 97.7, 97.6, 97.1, 97.2, 97.3, 96.9, 97.0, 96.9, 96.1, 95.6, 95.3, 94.7, 93.2, 91.9, 91.7, 90.7, 89.5, 86.7, 83.0]
                        }
plot_test_accuracy_coarse(accuracy_hist, accuracy_hist_subset)
#plot_test_accuracy_fine(accuracy_hist1, accuracy_hist2)