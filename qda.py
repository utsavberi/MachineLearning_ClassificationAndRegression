from __future__ import division
__author__ = 'utsav'


import pickle
import numpy as np
import matplotlib.pyplot as plt
import myPlot as mplt
import math

def qda_learn(X, y):
    classes = np.unique(y)
    Mean_ki = []
    # p = []

    c_list = []
    for i in classes:

        XNP=X[np.where(y==i)[0],]
        Mean_ki.append(np.mean(XNP,axis = 0))

        X_curr = X[np.where(y==i)[0],]
        c_list.append(np.cov(X_curr,rowvar=0))

    c_list = np.array(c_list)

    Mean_ki = np.array(Mean_ki)

    return  (Mean_ki,c_list)

# def qdaTest(means,covmat,Xtest,ytest):
#     classes = np.unique(ytest)
#     f = []
#     count = 0;
#
#     for i in classes:
#         C = covmat[count]
#         Cinv = np.matrix(C).I
#         ui = means[count]
#
#         z=(.5)*math.log(np.linalg.det(C)) - (.5)*( (Xtest-ui).dot(Cinv)).dot((Xtest-ui).T )# + (p[count])
#         f.append(z)
#         count += 1
#
#     f = np.array(f);
#     predicted = np.argmax(np.array(f),axis=0)[0]+1
#     # predicted = np.array(classes)[np.argmax(np.array(f),axis=0)[0]]
#     acc = np.mean((predicted.reshape(1,-1)[0]==ytest.reshape(1,-1)).astype(float)) * 100
#     return acc
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
     #trainingData is 150*2
    trainingData=X;
    rows=trainingData.shape[0];
    colums=trainingData.shape[1];
    #trueLabels is 150*1
    trueLables=y.reshape(y.size)
    #classLables will be in the range 1,2,3,4,5
    classLabels=np.unique(trueLables)
    #Means matrix dim will be 2*5
    means=np.zeros((colums,classLabels.size))
    #calculating the mean of the values where the classLabel=trueLabel
    covmats=[np.zeros((colums,colums))]*classLabels.size
    ##mean matrix is 2*5 one row represents x mean,other y mean
    for i in range(classLabels.size):
        means[:,i]=np.mean(trainingData[trueLables==classLabels[i]],axis=0)
        covmats[i]=np.cov(trainingData[trueLables==classLabels[i]],rowvar=0)


    return means,covmats
def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD
    pdf= np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        invcovmat = np.linalg.inv(covmats[i])
        covmatdet = np.linalg.det(covmats[i])
        pdf[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])*
        np.dot(invcovmat, (Xtest - means[:,i]).T).T,1))/(np.sqrt(np.pi*2)*(np.power(covmatdet,2)))
    #Getting the index of the class with the highest probability
    trueLabel = np.argmax(pdf,1)
    #Index start from 0,class index start from 1.So to balance the index adding 1 to all the index
    trueLabel = trueLabel + 1
    ytest = ytest.reshape(ytest.size)
    #calculating the accuracy
    acc = 100*np.mean(trueLabel == ytest)
    return acc

def qda_test():
    data = pickle.load( open( "sample.pickle", "rb" ) )
    xtrain =  data[0]
    ytrain = data[1]

    x = xtrain
    y = ytrain

    xtest = data[2]
    ytest = data[3]

    fit = qda_learn(x,y)

    print qdaTest(fit[0],fit[1],x,y)
    plt.show()


qda_test()