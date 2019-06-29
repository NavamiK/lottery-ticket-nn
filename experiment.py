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
    pruning_iterations = 27  # How many iterative pruning steps to perform

    print("Running a base experiment with input_size:{}, hidden_size:{}, num_classes:{}, "
          "batch_size:{}, num_epochs:{}, pruning_iterations:{}, pruning_rates:{}"
          .format(network.input_size, network.hidden_size, network.num_classes, network.batch_size, network.num_epochs,
                  pruning_iterations, network.pruning_rates ))

    train_loader, val_loader = dataset.init_data_mask_base_expt(n_train, n_valid)

    model = network.MultiLayerPerceptron().to(network.device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets

    network.train(model, train_loader, val_loader)

    test_accuracy_history = []
    test_accuracy = network.test(model)
    test_accuracy_history.append(test_accuracy)

    for iter in range(pruning_iterations):
        # This is the percentage of weights remaining in the network after pruning
        print("Results for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, 100 * 0.8**(iter + 1)))
        network.prune(model)
        network.train(model, train_loader, val_loader)
        test_accuracy = network.test(model)
        test_accuracy_history.append(test_accuracy)

    print('Test accuracy history {}'.format(test_accuracy_history))
    visualize.plot_test_accuracy(test_accuracy_history)

base_experiment()

def split_data_experiment():

    n_train1, n_train2 = 27500, 27500
    n_valid1, n_valid2 = 2500, 2500
    pruning_iterations = 1  # How many iterative pruning steps to perform

    train_loader1, val_loader1, train_loader2, val_loader2 = \
        dataset.init_data_mask_split_data_expt(n_train1, n_valid1, n_train2, n_valid2)

    model = network.MultiLayerPerceptron().to(network.device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets
    network.train(model, train_loader1, val_loader1)

    test_accuracy_history = []
    test_accuracy = network.test(model)
    test_accuracy_history.append(test_accuracy)

    for iter in range(pruning_iterations):
        # This is the percentage of weights remaining in the network after pruning
        print("Results for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, 100 * 0.8**(iter + 1)))
        network.prune(model)
        network.train(model, train_loader2, val_loader2)
        test_accuracy = network.test(model)
        test_accuracy_history.append(test_accuracy)

    print('Test accuracy history {}'.format(test_accuracy_history))
    visualize.plot_test_accuracy(test_accuracy_history)

#split_data_experiment()

print('Using device: %s'%network.device, ' end time: ' + str(datetime.datetime.now()))