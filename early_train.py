import argparse
import torch
from torchvision import datasets, transforms
import os


log_dir = '../data'

def save(file_name, data):
        file_name = os.path.join(log_dir, file_name)
        torch.save(data.cpu(), file_name)

def make_data():
    train_kwargs = {'batch_size': 5000, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    if not os.path.isdir(comp_dir):
        os.mkdir(comp_dir)

    for c in range(10):
        data_all = []
        for i, (x,y) in enumerate(train_loader):
            data = x[y == c]
            data_all.append(data)

        data = torch.concat(data_all)
        save(f'data_class_{i}.pt', data)

if __name__ == '__main__':
    make_random()
