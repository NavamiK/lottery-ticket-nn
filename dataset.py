import torchvision.transforms as transforms
import torchvision
import torch

num_training = 55000
num_validation = 5000
batch_size = 100

#-------------------------------------------------
# Load the MNIST dataset
#-------------------------------------------------
norm_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                     ])
mnist_dataset = torchvision.datasets.MNIST(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='datasets/',
                                          train=False,
                                          transform=norm_transform
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
data_mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(mnist_dataset, data_mask)
data_mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(mnist_dataset, data_mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
