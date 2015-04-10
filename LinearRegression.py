__author__ = 'utsav'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import myPlot as mplt
import numpy.linalg as n
import math


def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    #W= (X.t dot X)^-1 dot X.T Y
    # bias = np.ones(X.shape[0]).reshape((-1,1))
    # X = np.hstack((X,bias))
    #(XtX)^-1 XtY

    w = n.inv((X).T.dot(X)).dot((X).T).dot(y)
    print w
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:

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
W = learnOLERegression(xtrain,ytrain)



print testOLERegression(W,xtest,ytest)


# i = 0
# print xtrain[i]
# print W
#
# # bias = np.ones(xtrain.shape[0]).reshape((-1,1))
# # xtrain = np.hstack((xtrain,bias))
# print np.around(xtrain[i].dot(W))
# print ytrain[i]
#
# i = 1
# # print xtrain[i]
# # print W
# print np.around(xtrain[i].dot(W))
# print ytrain[i]
#
# i = 2
# # print xtrain[i]
# # print W
# print np.around(xtrain[i].dot(W))
# print ytrain[i]
#
# i = 3
# # print xtrain[i]
# # print W
# print np.around(xtrain[i].dot(W))
# print ytrain[i]