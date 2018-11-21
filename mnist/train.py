#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:41:27 2018

@author: 
"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import argparse

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class DynamicKernel(nn.Module):
    def __init__(self):
        super(DynamicKernel, self).__init__()
        self.k_size = 5
        self.k_radius = int(self.k_size/2)
        self.pad = self.k_radius
        self.img_size = 28 + self.k_size - 1
        self.batch = 64
        self.kernel = []
        self.dtype = torch.cuda.FloatTensor
        self.dtype_int = torch.cuda.LongTensor
        
        self.kernel = nn.Conv2d(1,self.k_size*self.k_size,self.k_size,1,
                                             (self.k_radius,self.k_radius)).type(self.dtype)
        
        self.h = torch.ones((self.k_size,self.k_size)).type(self.dtype_int)
        self.QConv = nn.Conv2d(1,1,self.k_size,stride=self.k_size).type(self.dtype)        
#        self.output = Variable(torch.zeros(1,1,self.img_size,self.img_size).type(self.dtype))
    
    def forward(self, x):
        x_tmp = F.pad(x,(self.pad,self.pad,self.pad,self.pad))
        output = Variable(torch.zeros(x.shape[0],1,self.img_size,self.img_size).type(self.dtype))
        firstConv = self.kernel(x_tmp)
        for i in range(self.k_size):
            for j in range(self.k_size):
                output += F.pad(x,(self.k_size-1-i,i,self.k_size-1-j,j))*torch.tanh(firstConv[:,self.k_size*i + j,:,:].view(x.shape[0],1,self.img_size,self.img_size))
                
        #return result without padding
        return output[:,:,self.pad:-self.pad,self.pad:-self.pad]

class DynamicLayer(nn.Module):
    def __init__(self):
        super(DynamicLayer,self).__init__()
        self.n_kernels_input = 1
        self.n_kernels_output = 6
        self.layer = []
        self.dtype = torch.cuda.FloatTensor
        for j in range(self.n_kernels_output):
            vartmp = []
            for i in range(self.n_kernels_input):
                vartmp.append(DynamicKernel())
            self.layer.append(vartmp)

    def forward(self,x):
        self.output = Variable(torch.zeros(x.shape[0],
                                           self.n_kernels_output,
                                           x.shape[2],
                                           x.shape[3]).type(self.dtype))
        for i in range(self.n_kernels_output):
            for j in range(self.n_kernels_input):
                img_tmp = self.layer[i][j](x[:,j,:,:].view(x.shape[0],1,x.shape[2],x.shape[3]))
                self.output[:,i,:,:] += img_tmp.view(x.shape[0],x.shape[2],x.shape[3])
        return self.output

import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quad1 = DynamicLayer()
      #  self.conv1 = nn.Conv2d(1, 10, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(6 , 50, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1250, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.quad1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1250)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

class tmp():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 30
        self.lr = 0.05
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10


args = tmp()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
count = 0
#for epoch in range(1, args.epochs + 1):
#    train(args, model, device, train_loader, optimizer, epoch)
#    test(args, model, device, test_loader)
    
lrs = [0.02,0.002,0.0002]
resumes = [False,True,True]
args = tmp()

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

for i in range(3):
  args.lr = lrs[i]
  args.resume = resumes[i]
  if args.resume:
      # Load checkpoint.
      print('==> Resuming from checkpoint..')
      assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
      checkpoint = torch.load('./checkpoint/ckpt.t7')
      model.load_state_dict(checkpoint['net'])
      best_acc = checkpoint['acc']
      start_epoch = checkpoint['epoch']

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=5e-4)

  # Training
  def train(epoch):
      print('\nEpoch: %d' % epoch)
      model.train()
      train_loss = 0
      correct = 0
      total = 0
      for batch_idx, (inputs, targets) in enumerate(train_loader):
          inputs, targets = inputs.to(device), targets.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()

      print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

  def test(epoch):
      global best_acc
      model.eval()
      test_loss = 0
      correct = 0
      total = 0
      with torch.no_grad():
          for batch_idx, (inputs, targets) in enumerate(test_loader):
              inputs, targets = inputs.to(device), targets.to(device)
              outputs = model(inputs)
              loss = criterion(outputs, targets)

              test_loss += loss.item()
              _, predicted = outputs.max(1)
              total += targets.size(0)
              correct += predicted.eq(targets).sum().item()

          print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

      # Save checkpoint.
      acc = 100.*correct/total
      if acc > best_acc:
          print('Saving..')
          state = {
              'net': model.state_dict(),
              'acc': acc,
              'epoch': epoch,
          }
          if not os.path.isdir('checkpoint'):
              os.mkdir('checkpoint')
          torch.save(state, './checkpoint/ckpt.t7')
          best_acc = acc


  for epoch in range(start_epoch, start_epoch+40):
      train(epoch)
      test(epoch) 
