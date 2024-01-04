from tqdm import trange
from time import sleep
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from functions import *
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class FAIR_HGR_NN(torch.nn.Module): 

    def __init__(self,regressor, mod_h,lr,p_device,nbepoch, lambdaHGR, nbepochHGR, start_epochHGR,mod_HGR_F,mod_HGR_G): 
        super().__init__()
        self.lr = lr
        self.device = torch.device(p_device)
        self.nbepoch = int(nbepoch)
        self.nbepochHGR = int(nbepochHGR)
        self.model_h = mod_h()
        self.lambdaHGR = lambdaHGR
        self.nbepochHGR = int(nbepochHGR)
        self.start_epochHGR = int(start_epochHGR)
        self.mF = mod_HGR_F()
        self.mG = mod_HGR_G()

        if regressor == 'mse':
          self.criterion = torch.nn.MSELoss(reduction='mean')
        elif regressor == 'rmse':
          self.criterion = RMSELoss()

        
    def predict(self, X_train): 
        x_var = Variable(torch.FloatTensor(X_train.values)).to(self.device)
        yhat= self.model_h(x_var)
        return yhat
    
    def fit(self, X_train, y_train, s_train): 
        
        self.optimizer_h = torch.optim.Adam(self.model_h.parameters(), lr=self.lr)
        self.model_h.to(self.device)

        self.optimizer_F = torch.optim.Adam(self.mF.parameters(), lr=self.lr)
        self.mF.to(self.device)
        
        self.optimizer_G = torch.optim.Adam(self.mG.parameters(), lr=self.lr)
        self.mG.to(self.device)     
        
        epsilon = 0.000000001

        loss=0
        ypred_var=0
        t = trange(self.nbepoch + 1, desc='Bar desc', leave=True)
        
        s_var = Variable(torch.FloatTensor(np.expand_dims(s_train,axis = 1))).to(self.device)
        x_var = Variable(torch.FloatTensor(X_train.values)).to(self.device)
        y_var = Variable(torch.FloatTensor(np.expand_dims(y_train,axis = 1))).to(self.device)
        
        for epoch in t: #tqdm(range(1, self.nbepoch + 1), 'Epoch: ', leave=False):

            # Mini batch learning
            t.set_description("Bar desc (file %i)" % epoch)
            t.refresh() # to show immediately the update            
            ret=0

            
            # Forward + Backward + Optimize
            if epoch < self.start_epochHGR:  ## Initial predictor model that makes sense before mitigating
                self.optimizer_h.zero_grad()
                ypred_var= self.model_h(x_var)
                loss = self.criterion(ypred_var, y_var)
                loss.backward()
                self.optimizer_h.step()
                y_pred_np=ypred_var.cpu().detach().numpy().squeeze(1)
                #print(y_pred_np[:5])
                #if epoch%5==0:
                #    print(rdc(y_pred_np, s_train))
            if epoch >= self.start_epochHGR:

                ypred_var0 = ypred_var.detach()

                for j in range(self.nbepochHGR) :

                    self.optimizer_F.zero_grad()
                    self.optimizer_G.zero_grad()

                    pred_F  = self.mF(ypred_var0)
                    pred_G  = self.mG(s_var)

                    pred_F_norm = (pred_F-torch.mean(pred_F))/torch.sqrt((torch.std(pred_F).pow(2)+epsilon))
                    pred_G_norm = (pred_G-torch.mean(pred_G))/torch.sqrt((torch.std(pred_G).pow(2)+epsilon))
                    #pred_F_norm[torch.isnan(pred_F_norm )] = 0
                    #pred_G_norm[torch.isnan(pred_G_norm )] = 0

                    ret = torch.mean(pred_F_norm*pred_G_norm)
                    lossHGR = - ret  # maximize

                    lossHGR.backward()

                    self.optimizer_F.step()
                    self.optimizer_G.step()

                self.optimizer_h.zero_grad()
                ypred_var= self.model_h(x_var)
                pred_F  = self.mF(ypred_var)
                pred_G  = self.mG(s_var)

                pred_F_norm = (pred_F-torch.mean(pred_F))/torch.sqrt((torch.std(pred_F).pow(2)+epsilon))
                pred_G_norm = (pred_G-torch.mean(pred_G))/torch.sqrt((torch.std(pred_G).pow(2)+epsilon))
                ret = torch.mean(pred_F_norm*pred_G_norm)
                #print('self.lambdaHGR*ret :',self.lambdaHGR*ret)
                #print('self.criterion(ypred_var, y_var)',self.criterion(ypred_var, y_var).cpu().detach().numpy() )
                loss = self.criterion(ypred_var, y_var) + self.lambdaHGR*ret #**2
                loss.backward()
                self.optimizer_h.step()

                y_pred_np=ypred_var.cpu().detach().numpy().squeeze(1)
                #if epoch%5==0:
                #    print(rdc(y_pred_np, s_train))

                    
        return y_pred_np #print('DONE')

