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
        # p.append(math.log(y[y==i].size / X.shape[0]))

        XNP=X[np.where(y==i)[0],]
        Mean_ki.append(np.mean(XNP,axis = 0))

        X_curr = X[np.where(y==i)[0],]
        c_list.append(np.cov(X_curr,rowvar=0))

    c_list = np.array(c_list)
    print "clist shape"
    print c_list.shape

    Mean_ki = np.array(Mean_ki)

    return  (Mean_ki,c_list)

def qdaTest(means,covmat,Xtest,ytest):
    classes = np.unique(ytest)
    f = []
    count = 0;

    for i in classes:
        C = covmat[count]
        Cinv = np.matrix(C).I
        ui = means[count]

        z=(.5)*math.log(np.linalg.det(C)) - (.5)*( (Xtest-ui).dot(Cinv)).dot((Xtest-ui).T )# + (p[count])
        f.append(z)
        count += 1

    f = np.array(f);
    # print np.array(f)
    print ytest.T
    # print np.array(classes)
    # la = np.argmax(np.array(f),axis=0)
    # print np.argmax(np.argmax(np.array(f),axis=0),axis=0)
    print np.array(classes)[np.argmax(f,axis=0)[0]]
    tmp = np.argmax(np.array(f),axis=0)
    predicted = np.array(classes)[np.argmax(np.array(f),axis=0)[0]]
    # print "shapes"
    # print predicted.reshape(1,-1)
    # print ytest.reshape(1,-1)
    acc = np.mean((predicted.reshape(1,-1)[0]==ytest.reshape(1,-1)).astype(float)) * 100
    return str(acc)+"%"
    # return "h"

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