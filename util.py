import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset

batch_size = 16
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
flipped_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=1.0)
    ]
)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_data():
    train_dataset = datasets.CIFAR10(root='./.data', train=True, download=True, transform=transform)
    flipped_train_dataset = datasets.CIFAR10(root='./.data', train=True, download=False, transform=flipped_transform)
    val_size = 5000

    train, val = random_split(train_dataset, [len(train_dataset) - val_size, val_size])
    train_loader = DataLoader(ConcatDataset([train, flipped_train_dataset]), batch_size=batch_size, num_workers=2)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=2)

    # mapping from label to english description, la
    classes = train_dataset.classes

    return train_loader, val_loader, classes


def get_test_data():
    test_dataset = datasets.CIFAR10(root='./.data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    # mapping from label to english description, la
    classes = test_dataset.classes

    return test_loader, classes


def get_save_path(model):
    if isinstance(model, SplitClassifier):
        return './trained_models/' + model.__class__.__name__ + '_' + str(model.splits) + '.pt'
    return './trained_models/' + model.__class__.__name__ + '.pt'
