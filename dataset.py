import torchvision.transforms as transforms
import torchvision
import torch

#num_training = 55000
#num_validation = 5000
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

def init_data_mask_base_expt(n_train, n_valid):
    # Prepare the training and validation splits
    train_data_mask = list(range(n_train))
    val_data_mask = list(range(n_train, n_train + n_valid))
    train_dataset = torch.utils.data.Subset(mnist_dataset, train_data_mask)
    val_dataset = torch.utils.data.Subset(mnist_dataset, val_data_mask)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def init_data_mask_split_data_expt(n_train1, n_valid1, n_train2, n_valid2):
    # Prepare the training and validation splits
    train_data_mask1 = list(range(n_train1))
    val_data_mask1 = list(range(n_train1, n_train1 + n_valid1))
    train_data_mask2 = list(range(n_train1 + n_valid1, n_train1 + n_valid1 + n_train2))
    val_data_mask2 = list(range(n_train1 + n_valid1 + n_train2, n_train1 + n_valid1 + n_train2 + n_valid2))

    train_dataset1 = torch.utils.data.Subset(mnist_dataset, train_data_mask1)
    val_dataset1 = torch.utils.data.Subset(mnist_dataset, val_data_mask1)
    train_dataset2 = torch.utils.data.Subset(mnist_dataset, train_data_mask2)
    val_dataset2 = torch.utils.data.Subset(mnist_dataset, val_data_mask2)

    # Data loader
    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=True)
    val_loader1 = torch.utils.data.DataLoader(dataset=val_dataset1, batch_size=batch_size, shuffle=False)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=True)
    val_loader2 = torch.utils.data.DataLoader(dataset=val_dataset2, batch_size=batch_size, shuffle=False)

    return train_loader1, val_loader1, train_loader2, val_loader2

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
