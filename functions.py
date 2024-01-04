

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import accuracy_score

from torch.autograd import Variable
import numpy
from math import pi, sqrt

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

from scipy.stats import rankdata



def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))






# Independence of 2 variables
def _joint_2(X, Y, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
    joint_density = density(data)

    nbins = int(min(50, 5. / joint_density.std))
    #nbins = np.sqrt( Y.size/5 )
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)

    xx, yy = torch.meshgrid([x_centers, y_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
    h2d = joint_density.pdf(grid) + damping
    h2d /= h2d.sum()
    return h2d


def hgr(X, Y, density, damping = 1e-10):
    """
    An estimator of the Hirschfeld-Gebelein-Renyi maximum correlation coefficient using Witsenhausen’s Characterization:
    HGR(x,y) is the second highest eigenvalue of the joint density on (x,y). We compute here the second eigenvalue on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: numerical value between 0 and 1 (0: independent, 1:linked by a deterministic equation)
    """
    h2d = _joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return torch.svd(Q)[1][1]


def chi_2(X, Y, density, damping = 0):
    """
    The \chi^2 divergence between the joint distribution on (x,y) and the product of marginals. This is know to be the
    square of an upper-bound on the Hirschfeld-Gebelein-Renyi maximum correlation coefficient. We compute it here on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: numerical value between 0 and \infty (0: independent)
    """
    h2d = _joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return ((Q ** 2).sum(dim=[0, 1]) - 1.)


# Independence of conditional variables

def _joint_3(X, Y, Z, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    Z = (Z - Z.mean()) / Z.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], -1)
    joint_density = density(data)  # + damping

    nbins = int(min(50, 5. / joint_density.std))
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)
    z_centers = torch.linspace(-2.5, 2.5, nbins)
    xx, yy, zz = torch.meshgrid([x_centers, y_centers, z_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], -1)

    h3d = joint_density.pdf(grid) + damping
    h3d /= h3d.sum()
    return h3d


def hgr_cond(X, Y, Z, density):
    """
    An estimator of the function z -> HGR(x|z, y|z) where HGR is the Hirschfeld-Gebelein-Renyi maximum correlation
    coefficient computed using Witsenhausen’s Characterization: HGR(x,y) is the second highest eigenvalue of the joint
    density on (x,y). We compute here the second eigenvalue on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param Z: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: A torch 1-D Tensor of same size as Z. (0: independent, 1:linked by a deterministic equation)
    """
    damping = 1e-10
    h3d = _joint_3(X, Y, Z, density, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return np.array(([torch.svd(Q[:, :, i])[1][1] for i in range(Q.shape[2])]))


def chi_2_cond(X, Y, Z, density):
    """
    An estimator of the function z -> chi^2(x|z, y|z) where \chi^2 is the \chi^2 divergence between the joint
    distribution on (x,y) and the product of marginals. This is know to be the square of an upper-bound on the
    Hirschfeld-Gebelein-Renyi maximum correlation coefficient. We compute it here on an empirical and discretized
    density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param Z: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: A torch 1-D Tensor of same size as Z. (0: independent)
    """
    damping = 0
    h3d = _joint_3(X, Y, Z, density, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return ((Q ** 2).sum(dim=[0, 1]) - 1.)


import torch
from math import pi, sqrt


class kde:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.train_x = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]
        assert d == self.d

        data = x.unsqueeze(-2)

        train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        pdf_values = (
                         torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
                     ).mean(dim=-1) / np.sqrt(2 * pi) / self.bandwidth

        return pdf_values


def _unsqueeze_multiple_times(input, axis, times):
    """
    Utils function to unsqueeze tensor to avoid cumbersome code
    :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
    :param axis: the axis to unsqueeze repeatedly
    :param times: the number of repetitions of the unsqueeze
    :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
    """
    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output





H=16
H2=8
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4    


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4
    
class Predictor(nn.Module):    
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(82, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.8)
        x = self.fc2(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.8)
        x = self.fc3(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.8)
        x = self.fc4(x)
        return x

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
H = 15
H2 = 15


class Net_HGR(nn.Module):
    def __init__(self):
        super(Net_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        return h4    


class Net2_HGR(nn.Module):
    def __init__(self):
        super(Net2_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        return h4    


model_Net_F = Net_HGR()
model_Net_G = Net2_HGR()


class HGR_NN(nn.Module):
    
    def __init__(self,model_F,model_G,device,display):
        super(HGR_NN, self).__init__()
        self.mF = model_Net_F
        self.mG = model_Net_G
        self.device = device
        self.optimizer_F = torch.optim.Adam(self.mF.parameters(), lr=0.0005)
        self.optimizer_G = torch.optim.Adam(self.mG.parameters(), lr=0.0005)
        self.display=display
    def forward(self, yhat, s_var, nb):

        svar = Variable(torch.FloatTensor(np.expand_dims(s_var,axis = 1))).to(self.device)
        yhatvar = Variable(torch.FloatTensor(np.expand_dims(yhat,axis = 1))).to(self.device)

        self.mF.to(self.device)
        self.mG.to(self.device)   
        
        for j in range(nb) :

            pred_F  = self.mF(yhatvar)
            pred_G  = self.mG(svar)
            
            epsilon=0.000000001
            
            pred_F_norm = (pred_F-torch.mean(pred_F))/torch.sqrt((torch.std(pred_F).pow(2)+epsilon))
            pred_G_norm = (pred_G-torch.mean(pred_G))/torch.sqrt((torch.std(pred_G).pow(2)+epsilon))

            ret = torch.mean(pred_F_norm*pred_G_norm)
            loss = - ret  # maximize
            self.optimizer_F.zero_grad()
            self.optimizer_G.zero_grad()
            loss.backward()
            
            if (j%100==0) and (self.display==True):
                print(j, ' ', loss)
            
            self.optimizer_F.step()
            self.optimizer_G.step()
            
        return ret.cpu().detach().numpy()
    

def FairQuant(s_test,y_test,y_predt_np):
    d = {'sensitivet': s_test, 'y_testt': y_test, 'y_pred3t':y_predt_np}
    df = pd.DataFrame(data=d)
    vec=[]
    for i in np.arange(0.02,1.02,0.02):
        tableq = df[df.sensitivet <= df.quantile(i)['sensitivet']]
        av_BIN  = tableq.y_pred3t.mean()
        av_Glob = df.y_pred3t.mean()
        vec=np.append(vec,(av_BIN-av_Glob))
    FairQuantabs50 = np.mean(np.abs(vec))
    FairQuantsquare50 = np.mean(vec**2)
    #print(FairQuantabs50)
    return FairQuantabs50
    
    