import torch
import torch.nn as nn
import datetime
import dataset
import visualize
import network

#--------------------------------
# Hyper-parameters
#--------------------------------


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


#--------------------------------
# Device configuration
#--------------------------------

print('Using device: %s'%network.device, ' start time: ' + str(datetime.datetime.now()))

def base_experiment():
    n_train, n_valid = 55000, 5000
    num_pruning_iter = 15  # How many iterative pruning steps to perform

    print("Running a base experiment with input_size:{}, hidden_size:{}, num_classes:{}, "
          "batch_size:{}, num_epochs:{}, num_pruning_iter:{}, pruning_rates:{}"
          .format(network.input_size, network.hidden_size, network.num_classes, network.batch_size, network.num_epochs,
                  num_pruning_iter, network.pruning_rates))

    train_loader, val_loader = dataset.init_data_mask_base_expt(n_train, n_valid)

    model = network.MultiLayerPerceptron().to(network.device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets

    early_stop_iteration = network.train(model, train_loader, val_loader)

    test_accuracy_history = []
    test_accuracy = network.test(model)
    test_accuracy_history.append(test_accuracy)
    early_stop_iteration_history = [early_stop_iteration]

    for iter in range(num_pruning_iter):
        # This is the percentage of weights remaining in the network after pruning
        print("\tResults for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, 100 * 0.8**(iter + 1)))
        network.prune(model)
        network.reset_params(model)
        early_stop_iteration = network.train(model, train_loader, val_loader)
        early_stop_iteration_history.append(early_stop_iteration)
        test_accuracy = network.test(model)
        test_accuracy_history.append(test_accuracy)

    print('Test accuracy history {}'.format(test_accuracy_history))
    print('Early stop iteration history {}'.format(early_stop_iteration_history))
    #visualize.plot_test_accuracy_coarse(test_accuracy_history)

#base_experiment()

def split_data_experiment():

    n_train1, n_train2 = 27500, 27500
    n_valid1, n_valid2 = 2500, 2500
    num_pruning_iter = 27  # How many iterative pruning steps to perform

    print("Running a data split experiment with input_size:{}, hidden_size:{}, num_classes:{}, "
          "batch_size:{}, num_epochs:{}, num_pruning_iter:{}, pruning_rates:{}"
          .format(network.input_size, network.hidden_size, network.num_classes, network.batch_size, network.num_epochs,
                  num_pruning_iter, network.pruning_rates))

    train_loader1, val_loader1, train_loader2, val_loader2 = \
        dataset.init_data_mask_split_data_expt(n_train1, n_valid1, n_train2, n_valid2)

    model = network.MultiLayerPerceptron().to(network.device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets
    network.train(model, train_loader1, val_loader1)

    test_accuracy1 = network.test(model)
    test_accuracy_history1 = [test_accuracy1]
    test_accuracy_history2 = [test_accuracy1]


    for iter in range(num_pruning_iter):
        # This is the percentage of weights remaining in the network after pruning
        print("Results for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, 100 * 0.8**(iter + 1)))
        # prune model after training with first half of data
        network.prune(model)
        # Reset, retrain on second half of data - perform testing
        network.reset_params(model)
        network.train(model, train_loader2, val_loader1)
        test_accuracy2 = network.test(model)
        test_accuracy_history2.append(test_accuracy2)
        # reset the model, retrain with first half of data - then perform testing
        network.reset_params(model)
        network.train(model, train_loader1, val_loader1)
        test_accuracy1 = network.test(model)
        test_accuracy_history1.append(test_accuracy1)

    print('Test accuracy history after re-training with first half of the training dataset {}'.format(test_accuracy_history1))
    print('Test accuracy history after re-training with second half of the training dataset {}'.format(
        test_accuracy_history2))
    visualize.plot_test_accuracy_coarse(test_accuracy_history1, test_accuracy_history2)

#split_data_experiment()


def subset_data_experiment(subset_size):

    n_train_full, n_valid_full = 55000, 5000
    n_train_subset, n_valid_subset = int(n_train_full * subset_size), int(n_valid_full * subset_size)
    num_pruning_iter = 15  # How many iterative pruning steps to perform

    print("Running a subset training experiment with subset_size:{}, input_size:{}, hidden_size:{}, num_classes:{}, "
          "batch_size:{}, num_epochs:{}, num_pruning_iter:{}, pruning_rates:{}"
          .format(subset_size, network.input_size, network.hidden_size, network.num_classes, network.batch_size, network.num_epochs,
                  num_pruning_iter, network.pruning_rates))

    train_loader_full, val_loader_full = dataset.init_data_mask_base_expt(n_train_full, n_valid_full)

    train_loader_subset, val_loader_subset = dataset.init_data_mask_base_expt(n_train_subset, n_valid_subset)

    model = network.MultiLayerPerceptron().to(network.device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets

    network.train(model, train_loader_subset, val_loader_subset)

    #test_accuracy_subset = network.test(model)
    test_accuracy_history_full = []
    test_accuracy_history_subset = []


    for iter in range(num_pruning_iter):
        # This is the percentage of weights remaining in the network after pruning
        print("\tResults for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, 100 * 0.8**(iter + 1)))
        # prune model after training with subset
        network.prune(model)
        # Reset, retrain the winning ticket on full dataset - perform testing
        network.reset_params(model)
        print("\tRetraining the winning ticket on whole dataset - to evaluate trainability and performance "
              "of sparse network ")
        network.train(model, train_loader_full, val_loader_full)
        test_accuracy_full = network.test(model)
        test_accuracy_history_full.append(test_accuracy_full)
        # reset the model, retrain with subset of data - to identify further winning tickets - then perform testing
        print("\tRetraining the winning ticket on subset of training dataset - to identify sparse network in next "
              "pruning cycle ")
        network.reset_params(model)
        network.train(model, train_loader_subset, val_loader_subset)
        test_accuracy_subset = network.test(model)
        test_accuracy_history_subset.append(test_accuracy_subset)

    print('Test accuracy history of winning ticket on subset of training data {}'.format(test_accuracy_history_subset))
    print('Test accuracy history of winning ticket after re-training with full training dataset {}'.format(
        test_accuracy_history_full))
    #visualize.plot_test_accuracy_coarse(test_accuracy_history1, test_accuracy_history2)

subset_size_to_evalute = [0.04, 0.1, 0.2, 0.4, 0.8]
for subset_size in subset_size_to_evalute:
    subset_data_experiment(subset_size)
    #pass

print('Using device: %s'%network.device, ' end time: ' + str(datetime.datetime.now()))