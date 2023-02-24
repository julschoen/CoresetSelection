import torch
from herding import Herding

class Trainer():
    def __init__(self, params, dataset):
        self.herder = Herding(dataset, params)

    def train(self):
        x = self.herder.select()
        for key in x:
            print(key)
            print(x[key])

