from torch import nn, optim
from torchvision import transforms, datasets

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ToRGB(object):
    """
    RGB transformer
    Takes 1 channel Tensor and return a 3 channel Tensor with same values
    for each channel
    """

    def __call__(self, tensor):
        return tensor.expand((3, tensor.shape[1], tensor.shape[2]))

class MNIST3D(torch.utils.data.Dataset):
    """MNIST dataset"""

    def __init__(self, root, transform='standard', upscaling_dim=(299, 299)):
        """
        Args
        root (str):                         The root path of the dataset
        transform (torchvision.transforms): Transformation pipeline to apply
        upscaling_dim (tuple):              The dimension of the upscaling
        """
        self.dim = upscaling_dim
        self.root = root
        self.transform = self.get_transform(transform)
        self.fetch_dataset()

    def get_transform(self, transform):
        """
        Get the chosen transformer
        """
        if transform == 'standard':
            return transforms.Compose([
                transforms.ToTensor(),
                ToRGB()
                #transforms.Normalize((0.1307,), (0.3081,)),
                #AddGaussianNoise(0, 0.2),
            ])

        elif transform == 'upscale':
            return transforms.Compose([
                transforms.Resize(self.dim, PIL.Image.LANCZOS),
                transforms.ToTensor(),
                ToRGB()
            ])
            
        elif type(transform) == torch.transforms.Compose:
            return transform


        else:
            raise NotImplementedError('Chosen transformer is not valid')

    def fetch_dataset(self):
        """
        Download MNIST dataset
        """
        self.train_dataset = datasets.MNIST(
            root=self.root,  # Define where dataset must be stored
            train=True,  # Retrieve training partition
            download=True,  # Retrieve dataset from remote repo
            transform=self.transform  # Apply chosen transforms
        )
        self.test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size, num_workers):
        """
        Get DataLoader instance from Dataset instance
        """
        # Get test dataset DataLoader object
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # Get train dataset DataLOader object
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        # Set training data loader
        return train_loader, test_loader

    def get_examples(self, loader, example_number = 0, n_examples = 1):
        running_n = 0
        for batch in loader:
            l = batch[0].shape[0]
            running_n += l
            if running_n > example_number:
                pos = example_number%l
                return batch[0][pos:pos+n_examples, ...], batch[1][pos:pos+n_examples, ...]

class MNIST(torch.utils.data.Dataset):
    """MNIST dataset"""

    def __init__(self, root, transform='standard', upscaling_dim=(299, 299)):
        """
        Args
        root (str):                         The root path of the dataset
        transform (torchvision.transforms): Transformation pipeline to apply
        upscaling_dim (tuple):              The dimension of the upscaling
        """
        self.dim = upscaling_dim
        self.root = root
        self.transform = self.get_transform(transform)
        self.fetch_dataset()

    def get_transform(self, transform):
        """
        Get the chosen transformer
        """
        if transform == 'standard':
            return transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,)),
                #AddGaussianNoise(0, 0.2),
            ])

        elif transform == 'upscale':
            return transforms.Compose([
                transforms.Resize(self.dim, PIL.Image.LANCZOS),
                transforms.ToTensor(),
                ToRGB()
            ])
        elif type(transform) == torch.transforms.Compose:
            return transform


        else:
            raise NotImplementedError('Chosen transformer is not valid')

    def fetch_dataset(self):
        """
        Download MNIST dataset
        """
        self.train_dataset = datasets.MNIST(
            root=self.root,  # Define where dataset must be stored
            train=True,  # Retrieve training partition
            download=True,  # Retrieve dataset from remote repo
            transform=self.transform  # Apply chosen transforms
        )
        self.test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size, num_workers):
        """
        Get DataLoader instance from Dataset instance
        """
        # Get test dataset DataLoader object
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # Get train dataset DataLOader object
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        # Set training data loader
        return train_loader, test_loader

    def get_examples(self, loader, example_number = 0, n_examples = 1):
        running_n = 0
        for batch in loader:
            l = batch[0].shape[0]
            running_n += l
            if running_n > example_number:
                pos = example_number%l
                return batch[0][pos:pos+n_examples, ...], batch[1][pos:pos+n_examples, ...]
