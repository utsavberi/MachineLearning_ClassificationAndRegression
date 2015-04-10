__author__ = 'utsav'

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from numpy.linalg import inv, det
import scipy.io
import matplotlib.pyplot as plt
import pickle

x  = np.array([[1],[2],[3],[4],[5],[6]])



def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    x = x.reshape(-1,1)
    print x.size
    tmp = np.ones((x.size,1))
    for i in range(1,p):
        tmp = np.hstack((tmp,np.power(x,i)))

    return tmp


X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
print X[:,2].reshape(1,-1)
print mapNonLinear(x,3)
#
# # Problem 5
pmax = 7
# lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    # Xdtest = mapNonLinear(Xtest[:,2],p)
    # w_d1 = learnRidgeRegression(Xd,y,0)
    # rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    # w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    # rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
# plt.plot(range(pmax),rmses5)
# plt.legend(('No Regularization','Regularization'))
