__author__ = 'utsav'

__author__ = 'utsav'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import myPlot as mplt
import numpy.linalg as n
import math

def learnRidgeERegression(X,y,lambd):
    N =  y.size
    I = np.eye(X.shape[1])
    w = n.inv((X).T.dot(X)+lambd*N*I).dot((X).T).dot(y)
    print w
    return w

def learnOLERegression(X,y):


    w = n.inv((X).T.dot(X)).dot((X).T).dot(y)
    print w
    return w

def testOLERegression(w,Xtest,ytest):

    Z = ytest-(Xtest.dot(w))
    Z = Z.T.dot(Z)
    N =  ytest.size

    rmse = math.sqrt( Z[0][0]) / N

    return rmse

data = pickle.load( open( "diabetes.pickle", "rb" ) )
xtrain =  data[0]
ytrain = data[1]

xtest = data[2]
ytest = data[3]
W = learnRidgeERegression(xtrain,ytrain,.3)

print testOLERegression(W,xtest,ytest)
