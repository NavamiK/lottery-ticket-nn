import torch
import torch.nn as nn
import torchvision
import numpy as np
import dataset

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 28 * 28
hidden_size = [300, 100]
num_classes = 10
batch_size = 100
# Set epoch such that num_epochs * num_iterations_per_epoch (num_training/batch_size) is the iteration limit
# as per paper - 50,000
num_epochs = 40
learning_rate = 1e-3
learning_rate_decay = 0.95
reg = 0.001

pruning_rates = {'layers.0.weight': 0.2,
                 'layers.2.weight': 0.2,
                 'layers.4.weight': 0.1}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#-------------------------------------------------
# Fully connected neural network with one hidden layer
#-------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[0], hidden_size[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[1], num_classes))
        self.layers = nn.Sequential(*layers)
        masks = {}

        for name, param in self.state_dict().items():
            if 'weight' in name:
                masks[name] = np.ones(np.prod(param.size()))
        self.masks = masks

    def forward(self, x):
        for name, param in self.named_parameters():
            if 'weight' in name:
                mask_tensor = torch.from_numpy(self.masks[name]).to(device, torch.float).reshape(param.size())
                param.data = param.data * mask_tensor  # flatten
        # for name, param in self.named_parameters():
        #     if 'layers.4.weight' in name:
        #         print('Weights for output layer in forward \n', param.data)

        out = self.layers(x.view(batch_size, input_size))
        return out


def validate(model, val_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_output = model.forward(val_images)
            predicted = torch.argmax(val_output, dim=1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
        valid_accuracy = 100 * correct / total
        #print('Validation accuracy is: {} %'.format(100 * correct / total))
    return valid_accuracy


def train(model, train_loader, val_loader):
    best_model = None
    best_valid_accuracy = 0
    val_acc_history = []

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=reg) # UPDATE - move this outside
    # Train the model
    lr = learning_rate
    iterations_per_epoch = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            output = model.forward(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_valid_accuracy = validate(model, val_loader)
        # Store validation accuracy history for plotting
        val_acc_history.append(current_valid_accuracy)

        # Find and save best model
        if current_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = current_valid_accuracy
            best_model = {
                'iteration': (epoch + 1) * iterations_per_epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'accuracy': best_valid_accuracy
            }
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)
    print("\t\tValidation accuracy history: {}, \tEarly stopping iteration: {}"
          .format(val_acc_history, best_model['iteration']))
    model.load_state_dict(best_model['model_state_dict'])
    return best_model['iteration']

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
        print('\t\tAccuracy of the network on the {} test images: {} %'.format(total, test_accuracy))
        return test_accuracy

def prune(model):
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if 'weight' in name:
            weights = param.cpu().numpy().reshape(-1)
            sorted_weights = np.sort(np.abs(weights[model.masks[name] == 1]))
            cutoff_index = np.round(pruning_rates[name] * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index]
            model.masks[name] = np.where(np.abs(weights) <= cutoff, np.zeros(model.masks[name].shape),
                                         model.masks[name])
            # if 'layers.4.weight' in name:
            # print('Trained weights for output layer\n', param)
            # print('Mask for output layer\n', model.masks[name])

def reset_params(model):
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if 'weight' in name:
            state_dict[name].copy_(model.presets[name])
