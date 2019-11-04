#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import torch

class RedeNeualConvolucional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.numero_classes = 10

        self.conv1 =  torch.nn.Conv2d(1,20,5,1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x,2,2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x,2,2)
        x = x.view(-1, 4*4*50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)