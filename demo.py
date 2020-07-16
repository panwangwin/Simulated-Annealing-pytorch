# -*- coding:utf-8 -*-
# Created at 2020-07-16
# Filename:demo.py
# Author:Wang Pan
# Purpose:

import numpy as np
import torch
import torch.nn as nn
from Simulated_Annealing_Optimizer import SimulatedAnealling
import argparse
import matplotlib.pyplot as plt

def demo_function_np(x):
    f=(1/10)*(x-2)*(x-1)*(x+1)*(x+2)*(x+3)*(x+4)
    return f


class demo_function(nn.Module):
    def __init__(self):
        super(demo_function,self).__init__()
        self.x=nn.Parameter(torch.tensor([-1]).float())

    def forward(self):
        x=self.x
        f=1/10*(x-2)*(x-1)*(x+1)*(x+2)*(x+3)*(x+4)
        return f



class handler():
    def __init__(self,model,args):
        self.model=model
        self.loss_fn=self.set_loss(args.loss_fn)
        self.y=torch.tensor(-10000).float()              # I want to find the minimum of my function so I set target low
        self.epochs=args.epochs
        self.optimizer=SimulatedAnealling(params=model.parameters(),init_temp=args.init_temp,cooling=args.cooling
                                          ,lr=args.lr,model=self.model,loss_fn=self.loss_fn,y=self.y)

    @staticmethod
    def set_loss(loss_name):
        if loss_name == 'MSELoss':
            return nn.MSELoss()
        elif loss_name == 'L1Loss':
            return nn.L1Loss()

    def find_optimal(self):
        for _ in range(self.epochs):
            l=self.optimizer.step()
            print(l)

    def watcher(self):
        x=self.model.x.detach().numpy()
        y=self.model().detach().numpy()
        plt.scatter(x,y)
        x=np.arange(-3.5,2.1,0.01)
        y=demo_function_np(x)
        plt.plot(x,y)
        plt.show()




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--init_temp',default=10000,type=int)
    parser.add_argument('--cooling',default=2,type=int)
    parser.add_argument('--lr',default=0.1)
    parser.add_argument('--loss_fn',default='L1Loss')
    parser.add_argument('--epochs',default=20)
    args=parser.parse_args()
    kernel=demo_function()
    bar=handler(kernel,args)
    bar.find_optimal()
