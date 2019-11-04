#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import torch

class RedeTotalmenteConectada(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.numero_classes = 10

        self.layer_1 = torch.nn.Linear(784,100)
        self.layer_2 = torch.nn.Linear(100,10)

        self.activation = torch.nn.ReLU()
    
    def forward(self,x):
        first_layer = self.activation(self.layer_1(x))
        second_layer = self.activation(self.layer_2(first_layer))
        return second_layer