# -*- coding:utf-8 -*-
# Created at 2020-07-16
# Filename:Simulated_Annealing_Optimizer.py
# Author:Wang Pan
# Purpose:

import torch
import torch.optim as optim
import copy

class SimulatedAnealling(optim.Optimizer):
    def __init__(self,params,init_temp,cooling,lr,model,loss_fn,y):
        '''
        :param params: model_params
        :param init_temp: initial temperature
        :param cooling: cooling rate for each step
        :param lr: learning rate
        :param model: model needs to be optimized
        :param loss_fn: loss function
        :param y: target value
        '''
        defaults=dict(temp=init_temp,cooling=cooling,lr=lr,model=model,loss_fn=loss_fn,y=y)
        super(SimulatedAnealling,self).__init__(params,defaults)

    def step(self):
        for group in self.param_groups:
            temp=group['temp']
            cooling=group['cooling']
            lr=group['lr']
            model=group['model']
            loss_fn=group['loss_fn']
            y=group['y']
            result=model()
            loss1=loss_fn(result,y)
            old_params=copy.deepcopy(group['params'])  # store old params
            for p in group['params']:
                d_p=torch.rand(p.shape)
                p.data=p.data+d_p*lr                   # update params
            new_result=model()
            loss2=loss_fn(new_result,y)
            loss=loss2
            if ((loss2>loss1) and (torch.Tensor(1).uniform_()>((loss1 - loss2) / temp).exp())):
                loss=loss1
                for i,each in enumerate(group['params']):
                    each.data=old_params[i].data
            group['temp']=temp/cooling                 # cooling
        return loss