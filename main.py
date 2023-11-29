
## Standard libraries
import math
import numpy as np
import random

## Imports for plotting
import matplotlib.pyplot as plt

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# reproducibility
pl.seed_everything(42)
torch.manual_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



## this is the main file for running the code
from data_setup import get_data, set_cl_data
from visualization import plot_thermo
from losses import compute_loss , Trainer
from LangevianMC import Sampler
from model import Networks





# set up
def set_up(optimizer_type, model_size, model_type, batch_size,learning_rate, from_scrach):
           model = Networks(model_size, model_type).to(device)
           sampler_m = Sampler(model,mode='marginal', img_shape = (1,28,28), sample_size = 60 ,from_scrach=from_scrach,max_len=8192)
           sampler_j = Sampler(model,mode='joint'   , img_shape = (1,28,28), sample_size = 60 ,from_scrach=from_scrach, max_len=8192)

           if optimizer_type =='sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum =0.9)
           elif optimizer_type=='adam':
             optimizer = optim.Adam(model.parameters(), lr=learning_rate)

           crosEn = nn.CrossEntropyLoss()
           return model, sampler_m, sampler_j, optimizer, crosEn


# Hyper parameters


# Data
labels =[0,1,2,3,4,5,6,7,8,9];  # we can exclude some labels to study change in information content of the model
batch_size =6; 
data_size = 60000 ;

#type of problem
cl_type = 'm_ebm'
noise  =  0
model_size = 32
model_type = 'shallow'
gamma = 1 # the trade-off between joint and marginal loss (one equate the gcl loss to cross entropy loss)


# Hyper learning
epochs =20 ;  learning_rate = 1e-4 ; from_scrach = 0.05
optimizer_type = 'adam'

# sampling
MC_steps = 30
sample_size = batch_size


## load data
train_set , test_set = get_data( labels = labels, rotate = False , degree = 0)

train_set, _ =  torch.utils.data.random_split(train_set, [ data_size, train_set.data.shape[0] - data_size])
# classifier_loader = set_cl_data(labels , expert_size,train_set)
train_loader   = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=2, pin_memory=True)
test_loader    = data.DataLoader(test_set , batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=2, pin_memory=True)
test_sample    = next( iter ( test_loader))
#set up
model, sampler_m, sampler_j, optimizer, crosEn = set_up(optimizer_type , model_size, model_type, batch_size = sample_size,learning_rate = learning_rate , from_scrach=from_scrach)
losses = compute_loss( model = model , sampler_j = sampler_j , sampler_m = sampler_m, steps= MC_steps, noise  = noise )

trainer = Trainer(cl_type)

for epoch in range(epochs):

  print(epoch)
  model = trainer.learn(model,gamma)


# plot themodynamics quantities - phase=True markes possible qunatum phase transition at point cc 
plot_thermo(phase=False, cc=0 ,t=-1)


if __name__ == "__main__":
    pass
  




