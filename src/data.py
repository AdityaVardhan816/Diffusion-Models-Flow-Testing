"""Shared data loading pipeline for Diffusion and Flow Matching experiments."""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataset(name: str, root: str = "./data", train: bool = True):
    """Load dataset with standardized normalization to [-1, 1]."""
    os.makedirs(root, exist_ok=True)
    name = name.lower()

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    elif name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)

    elif name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: 'cifar10', 'fashion_mnist', 'mnist'.")


def get_dataloader(
    dataset_name: str = "cifar10",
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    train: bool = True,
    max_samples: int = None,
    seed: int = 42,
) -> DataLoader:
    """Create a reproducible DataLoader with shared preprocessing."""
    dataset = get_dataset(dataset_name, root=root, train=train)

    if max_samples is not None and max_samples < len(dataset):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
