import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class CIFAR10Dataset(Dataset):
    def __init__(self, cifar10_dataset, indices):
        self.cifar10_dataset = cifar10_dataset
        self.indices = indices

    def __make_class_indices(self):
        

    def __getitem__(self, idx):
        image, label = self.cifar10_dataset[self.indices[idx]]
        return image, label

    def __len__(self):
        return len(self.indices)

class CIFAR10DataLoader(DataLoader):
    def __init__(self, cifar10_dataset, num_samples_per_class, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False):
        self.cifar10_dataset = cifar10_dataset
        self.num_samples_per_class = num_samples_per_class
        self.class_indices = self._get_class_indices()
        self.indices = self._generate_indices()
        super().__init__(CIFAR10Dataset(self.cifar10_dataset, self.indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last)

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.cifar10_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _generate_indices(self):
        indices = []
        for label in self.class_indices:
            indices.extend(torch.randperm(len(self.class_indices[label]))[:self.num_samples_per_class])
        return indices