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
import matplotlib.animation as animation

def demo_function_np(x):
    f=(1/10)*(x-2)*(x-1)*(x+1)*(x+2)*(x+3)*(x+4)
    return f


class demo_function(nn.Module):
    def __init__(self,init_var):
        super(demo_function,self).__init__()
        self.x=nn.Parameter(torch.tensor([init_var]).float())

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
        self.x=[]
        self.y=[]

    @staticmethod
    def set_loss(loss_name):
        if loss_name == 'MSELoss':
            return nn.MSELoss()
        elif loss_name == 'L1Loss':
            return nn.L1Loss()

    def find_optimal(self):
        for _ in range(self.epochs):
            l=self.optimizer.step()
            self.x.append(self.model.x.detach().numpy()[0])
            self.y.append(self.model().detach().numpy()[0])
            print(l)
        self.watcher()


    def watcher(self):
        # Plot function
        x=np.arange(-4.3,2.3,0.01)
        y=demo_function_np(x)
        # Plot update animation
        fig, ax = plt.subplots()
        ax.plot(x,y)
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'ro', animated=True)

        def init():
            return ln,

        def update(frame):
            xdata.append(frame[0])
            ydata.append(frame[1])
            ln.set_data(xdata, ydata)
            return ln,

        def data_gen(epochs):
            i=0
            while i<epochs:
                yield (self.x[i],self.y[i])
                i=i+1

        anim = animation.FuncAnimation(fig, update, frames=data_gen(self.epochs), interval=10,
                                       init_func=init,repeat_delay=200)
        anim.save('./fig/test_animation.gif', writer='pillow')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--init_temp',default=10000)
    parser.add_argument('--cooling',default=1.5)
    parser.add_argument('--lr',default=1)
    parser.add_argument('--loss_fn',default='L1Loss')
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--init_var',default=-1)
    args=parser.parse_args()
    kernel=demo_function(args.init_var)
    bar=handler(kernel,args)
    bar.find_optimal()
