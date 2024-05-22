from pathlib import Path
import requests
import pickle
import gzip
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms, datasets


def x_transform_to_vec(x):
    trans = transforms.Compose([transforms.ToTensor()])
    x = trans(x)
    return x.view(-1)


def load_mnist(batch_size, n_class, shape=None):
    # download mnist data if it does not exist in "data/mnist"
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    # load mnist data
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    # convert mnist data to torch.tensor
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n_train, d = x_train.shape
    n_test, d = x_valid.shape

    center_img = torch.mean(x_train, dim=0, keepdim=True)
    center_pixel = torch.mean(x_train, dim=(0, 1), keepdim=True)
    center_zero = torch.tensor(0)

    if shape:
        x_train = x_train.view(-1, 1, 28, 28)
        x_valid = x_valid.view(-1, 1, 28, 28)
        
    # preprocessing the training and test data
    y_train = y_train % n_class
    y_valid = y_valid % n_class

    
    # set up data loader
    train_data = TensorDataset(x_train, y_train)
    train_data_loader_batch = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)
    train_data_loader_full = DataLoader(train_data, batch_size=n_train,num_workers=8)

    test_data = TensorDataset(x_valid, y_valid)
    test_data_loader_batch = DataLoader(test_data, batch_size=batch_size,num_workers=8, shuffle=False)
    test_data_loader_full = DataLoader(test_data, batch_size=n_test,num_workers=8, shuffle=False)

    return train_data_loader_batch, train_data_loader_full, test_data_loader_batch, test_data_loader_full, [center_img,
                                                                                                            center_pixel,
                                                                                                            center_zero], n_train, n_test, d


def load_fmnist(batch_size, n_class):
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=x_transform_to_vec,
                                                     target_transform=lambda y: y % n_class, download=False)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=x_transform_to_vec,
                                                       target_transform=lambda y: y % n_class, download=False)

    n_train = len(training_set)
    n_test = len(validation_set)

    train_data_loader_full = torch.utils.data.DataLoader(training_set, batch_size=n_train, shuffle=True)
    test_data_loader_full = torch.utils.data.DataLoader(validation_set, batch_size=n_test, shuffle=False)

    x_train, y_train = next(iter(train_data_loader_full))
    n_train, d = x_train.shape
    center_img = torch.mean(x_train, dim=0, keepdim=True)
    center_pixel = torch.mean(x_train, dim=(0, 1), keepdim=True)
    center_zero = torch.tensor(0)

    train_data_loader_batch = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_data_loader_batch = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    return train_data_loader_batch, train_data_loader_full, test_data_loader_batch, test_data_loader_full, [center_img,
                                                                                                            center_pixel,
                                                                                                            center_zero], n_train, n_test, d


def load_kmnist(batch_size, n_class):
    training_set = torchvision.datasets.KMNIST('./data', train=True, transform=x_transform_to_vec,
                                               target_transform=lambda y: y % n_class, download=False)
    validation_set = torchvision.datasets.KMNIST('./data', train=False, transform=x_transform_to_vec,
                                                 target_transform=lambda y: y % n_class, download=False)

    n_train = len(training_set)
    n_test = len(validation_set)

    train_data_loader_full = torch.utils.data.DataLoader(training_set, batch_size=n_train, shuffle=True)
    test_data_loader_full = torch.utils.data.DataLoader(validation_set, batch_size=n_test, shuffle=False)

    x_train, y_train = next(iter(train_data_loader_full))
    n_train, d = x_train.shape
    center_img = torch.mean(x_train, dim=0, keepdim=True)
    center_pixel = torch.mean(x_train, dim=(0, 1), keepdim=True)
    center_zero = torch.tensor(0)

    train_data_loader_batch = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_data_loader_batch = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    return train_data_loader_batch, train_data_loader_full, test_data_loader_batch, test_data_loader_full, [center_img,
                                                                                                            center_pixel,
                                                                                                            center_zero], n_train, n_test, d


if __name__ == '__main__':
    pass
