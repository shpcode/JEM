from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils import data
import torch
import numpy as np

DATASET_PATH = "path/to/dataset"  # Specify the actual path to your dataset

def get_data(labels, rotate, degree):
    """
    Get training and testing data from MNIST dataset with specified labels.

    Parameters:
    - labels (list): List of labels to include in the dataset.
    - rotate (bool): If True, apply random rotation to the test set.
    - degree (int): Rotation degree if rotate is True.

    Returns:
    - train_set (torch.utils.data.Dataset): Training dataset.
    - test_set (torch.utils.data.Dataset): Testing dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if rotate:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomRotation((-degree, degree), fill=-1)
        ])
    else:
        test_transform = transform

    train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    test_set = MNIST(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    train_dex = 0
    test_dex = 0
    for l in labels:
        train_dex = train_dex + (train_set.targets == l)
        test_dex = test_dex + (test_set.targets == l)

    train_set.data = train_set.data[train_dex == 1]
    train_set.targets = train_set.targets[train_dex == 1]
    test_set.targets = test_set.targets[test_dex == 1]
    test_set.data = test_set.data[test_dex == 1]

    new_l = 0
    for l in labels:
        train_set.targets[train_set.targets == l] = new_l
        test_set.targets[test_set.targets == l] = new_l
        new_l = new_l + 1

    return train_set, test_set


def set_cl_data(labels, expert_size, train_set):
    ''' 
    This function initializes a DataLoader for the classifier model. 
    The expert_size parameter determines the size of each subset of data.
    '''

    dex = []
    labels = list(np.arange(len(labels)))
    for l in labels:
        label_dex = torch.where(train_set.targets == l)
        rand_dex = torch.randint(0, label_dex[0].shape[0], (expert_size, 1))
        dex.append([(label_dex[0][rand_dex]).numpy()])

    dex = np.asarray(dex).reshape(expert_size * len(labels))
    classifier_data = torch.utils.data.Subset(train_set, dex)

    batch_size = expert_size if expert_size < 100 else 50

    classifier_loader = data.DataLoader(
        classifier_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True
    )
    return classifier_loader
