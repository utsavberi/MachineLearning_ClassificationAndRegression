from __future__ import division
__author__ = 'utsav'


import pickle
import numpy as np
import matplotlib.pyplot as plt
import myPlot as mplt
import math

def lda_learn(X, y):
    classes = np.unique(y)
    Mean_ki = []
    # Mean_global = np.mean(X,axis = 0)
    for i in classes:
        XNP=X[np.where(y==i)[0],]
        Mean_ki.append(np.mean(XNP,axis = 0))

    Mean_ki = np.array(Mean_ki)

    C = np.cov(X.T)
    print "cov"
    print C
    print np.cov(X ,rowvar=0).shape
    return (Mean_ki, C)

def ldaTest(means,covmat,Xtest,ytest):

    print "covmat"
    print covmat.shape
    Cinv = np.matrix(covmat).I
    classes = np.unique(ytest)
    # p = []
    # for i in classes:
    #     p.append(ytest[ytest==i].size / Xtest.shape[0])
    f = []
    count = 0;
    for i in classes:
        ui = means[count]
        f.append(ui.dot(Cinv).dot(Xtest.T) - (.5) * ui.dot(Cinv).dot(ui.T))#+math.log(p[count]))
        # print (ui.dot(Cinv).dot(Xtest.T)).shape
        # print (Xtest.T).shape
        # print Cinv.shape
        # print ui.shape
        # print ui
        # print "dddd"
        # print f[0].shape
        count += 1

    print "f"
    print np.array(f).shape
    predicted = np.array(classes)[np.argmax(np.array(f),axis=0)[0]]
    print predicted
    acc = np.mean((predicted.reshape(1,-1)[0]==ytest.reshape(1,-1)).astype(float)) * 100
    return str(acc)+"%"

def lda_test():
    data = pickle.load( open( "sample.pickle", "rb" ) )
    xtrain =  data[0]
    ytrain = data[1]

    # print np.hstack((xtrain,ytrain))
    x = xtrain
    y = ytrain

    xtest = data[2]
    ytest = data[3]

    # mplt.plot_classes(xtest,ytest)

    fit = lda_learn(x,y)

    print ldaTest(fit[0],fit[1],xtest,ytest)
    plt.show()


lda_test()