from __future__ import division
__author__ = 'utsav'


import pickle
import numpy as np
import matplotlib.pyplot as plt
import myPlot as mplt

def lda_learn(X, y):
    classes = np.unique(y)
    Mean_ki = []
    Mean_global = np.mean(X,axis = 0)
    for i in classes:
        XNP=X[np.where(y==i)[0],]
        Mean_ki.append(np.mean(XNP,axis = 0))

    Mean_ki = np.array(Mean_ki)

    c_list = []
    for i in classes:
        X_curr = X[np.where(y==i)[0],]
        X_mean_corrected = X_curr - Mean_global
        c_list.append(X_mean_corrected.T.dot(X_mean_corrected) / X_curr.shape[0])

    c_list = np.array(c_list)

    C = 1./X.shape[0]

    c_sum = np.zeros(c_list[0].shape)
    for i in classes:
         X_curr = X[np.where(y==i)[0],]
         c_sum += X_curr.shape[0] * c_list[i-1]

    C = C * c_sum

    p = []
    for i in classes:
        p.append(y[y==i].size / X.shape[0])

    # print "u"
    # print Mean_global
    # print "means"
    # print Mean_ki
    # print "covariance"
    # print C
    # print "p"
    # print p
    return (Mean_ki,C)#,p)

def ldaTest(means,covmat,Xtest,ytest):
    Cinv = np.matrix(covmat).I
    classes = np.unique(ytest)
    f = []
    count = 0;
    for i in classes:
        ui = means[count]
        f.append(ui.dot(Cinv).dot(Xtest.T) - (.5) * ui.dot(Cinv).dot(ui.T))#+math.log(p[count]))
        count += 1

    print "f"
    print f
    predicted = np.array(classes)[np.argmax(np.array(f),axis=0)[0]]
    acc = np.mean((predicted.reshape(1,-1)[0]==ytest.reshape(1,-1)).astype(float)) * 100
    return str(acc)+"%"

def lda_test():
    data = pickle.load( open( "sample.pickle", "rb" ) )
    xtrain =  data[0]
    ytrain = data[1]
    x = xtrain
    y = ytrain

    xtest = data[2]
    ytest = data[3]

    mplt.plot_classes(xtest,ytest)

    fit = lda_learn(x,y)

    print ldaTest(fit[0],fit[1],xtest,ytest)
    plt.show()


lda_test()