import torch
import torch.nn as nn
import torchvision
import sys
import datetime
import numpy as np
import collections
import dataset
import visualize

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device, ' start time: ' + str(datetime.datetime.now()))

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 28 * 28
#hidden_size = [300, 100]
hidden_size = [30, 10]
num_classes = 10
# Set epoch such that num_epochs * num_iterations_per_epoch (num_training/batch_size) is the iteration limit
# as per paper - 50,000
num_epochs = 2
pruning_iterations = 20  # How many iterative pruning steps to perform
batch_size = 100
learning_rate = 1e-3
learning_rate_decay = 0.95
reg=0.001
train = True

# weights_size = {'layers.0.weight': hidden_size[0] * input_size,
#                 'layers.2.weight': hidden_size[1] * hidden_size[0],
#                 'layers.4.weight': num_classes * hidden_size[1]}
pruning_rates = {'layers.0.weight': 0.2,
                 'layers.2.weight': 0.2,
                 'layers.4.weight': 0.1}
winning_ticket_weights = []


def zero_grad(self, grad_input, grad_output):
    print('grad_input: {}, grad_output: {}'.format(grad_input, grad_output))
    return grad_input * self.masks

#-------------------------------------------------
# Fully connected neural network with one hidden layer
#-------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[0], hidden_layers[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[1], num_classes))
        self.layers = nn.Sequential(*layers)
        masks = {}

        for name, param in self.state_dict().items():
            if 'weight' in name:
                masks[name] = np.ones(np.prod(param.size()))
        self.masks = masks
        #self.handle = self.register_backward_hook(zero_grad)
        #self.presets = presets

    def forward(self, x):
        for name, param in self.named_parameters():
            if 'weight' in name:
                mask_tensor = torch.from_numpy(self.masks[name]).to(device, torch.float).reshape(param.size())
                 #preset_tensor = model.presets[name].to(device)
                param.data = param.data * mask_tensor  # flatten
            # if 'layers.4.weight' in name:
            #     print('Weights for output layer in forward - immediate \n', param.data)
        # for name, param in self.named_parameters():
        #     if 'layers.4.weight' in name:
        #         print('Weights for output layer in forward \n', param.data)

        out = self.layers(x.view(batch_size, input_size))
        return out


def validate(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for val_images, val_labels in dataset.val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_output = model.forward(val_images)
            predicted = torch.argmax(val_output, dim=1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    print('Validation accuracy is: {} %'.format(100 * correct / total))

def train(model):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=reg) # UPDATE - move this outside
    # Train the model
    lr = learning_rate
    iterations_per_epoch = len(dataset.train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataset.train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            output = model.forward(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, iterations_per_epoch * num_epochs, loss.item()))
                validate(model)

        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)

def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataset.test_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(batch_size, input_size)
            output = model.forward(images)
            predicted = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total == 1000:
                break
        test_accuracy = 100 * correct / total
        print('Accuracy of the network on the {} test images: {} %'.format(total, test_accuracy))
        return test_accuracy

def prune(model):
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if 'weight' in name:

            weights = param.cpu().numpy().reshape(-1)
            sorted_weights = np.sort(np.abs(weights[model.masks[name] == 1]))
            cutoff_index = np.round(pruning_rates[name] * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index]
            model.masks[name] = np.where(np.abs(weights) <= cutoff, np.zeros(model.masks[name].shape), model.masks[name])
            #if 'layers.4.weight' in name:
                #print('Trained weights for output layer\n', param)

            # Moving below to forward pass
            #mask_tensor = torch.from_numpy(model.masks[name]).to(device, torch.float)
            #preset_tensor = model.presets[name].to(device)
            #winning_param = (preset_tensor.reshape(1, -1).squeeze() * mask_tensor).reshape(param.size())  # flatten

            #if 'layers.4.weight' in name:
                #print('Mask for output layer\n', model.masks[name])

            #if 'layers.4.weight' in name:
                #print('Winning ticket for output layer\n', winning_param)
            # Reset to original weights
            state_dict[name].copy_(model.presets[name])


def base_experiment():

    model = MultiLayerPerceptron(input_size, hidden_size, num_classes).to(device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets

    train(model)

    test_accuracy_history = []
    test_accuracy = test(model)
    test_accuracy_history.append(test_accuracy)

    sparsity_weights_history = [100]

    for iter in range(pruning_iterations):
        # This is the percentage of weights remaining in the network after pruning
        sparsity_weights = 100 * 0.8**(iter + 1)
        print("Results for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, sparsity_weights))
        prune(model)
        train(model)
        test_accuracy = test(model)
        sparsity_weights_history.append(sparsity_weights)
        test_accuracy_history.append(test_accuracy)
        visualize.plot_test_accuracy(sparsity_weights_history, test_accuracy_history)


def partial_dataset_experiment():

    model = MultiLayerPerceptron(input_size, hidden_size, num_classes).to(device)
    model.apply(weights_init)

    presets = {}
    for name, param in model.state_dict().items():
        presets[name] = param.clone()
    model.presets = presets

    train(model)

    test_accuracy_history = []
    test_accuracy = test(model)
    test_accuracy_history.append(test_accuracy)

    for iter in range(pruning_iterations):
        # This is the percentage of weights remaining in the network after pruning
        sparsity_weights = 100 * 0.8**(iter + 1)
        print("Results for pruning round {} with percentage of weights remaining {}"
              .format(iter + 1, sparsity_weights))
        prune(model)
        train(model)
        test_accuracy = test(model)
        sparsity_weights_history.append(sparsity_weights)
        test_accuracy_history.append(test_accuracy)
        visualize.plot_test_accuracy(sparsity_weights_history, test_accuracy_history)

base_experiment()
print('Using device: %s'%device, ' end time: ' + str(datetime.datetime.now()))