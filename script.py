import numpy as np
from scipy.optimize import minimize
from math import sqrt
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    classes = np.unique(y)
    means = []
    for i in classes:
        XNP=X[np.where(y==i)[0],]
        means.append(np.mean(XNP,axis = 0))
    means = np.array(means)
    covmat = np.cov(X.T)
    return means,covmat

def qdaLearn(X, y):
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

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    classes = np.unique(ytest)
    means = means.T
    z= np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        num = np.exp(-0.5*np.sum((Xtest - means[:,i])* np.dot(inv(covmats[i]), (Xtest - means[:,i]).T).T,1))
        denom = (np.sqrt(np.pi*2)*(np.power(det(covmats[i]),2)))
        z[:,i] = num / denom

    predicted = classes[np.argmax(z,1)]
    acc = np.mean((predicted.reshape(1,-1)[0]==ytest.reshape(1,-1)).astype(float)) * 100
    return acc



def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    Cinv = np.matrix(covmat).I
    classes = np.unique(ytest)
    f = []
    count = 0;
    for i in classes:
        ui = means[count]
        f.append(ui.dot(Cinv).dot(Xtest.T) - (.5) * ui.dot(Cinv).dot(ui.T))#+math.log(p[count]))
        count += 1

    predicted = np.array(classes)[np.argmax(np.array(f),axis=0)[0]]
    acc = np.mean((predicted.reshape(1,-1)[0]==ytest.reshape(1,-1)).astype(float)) * 100
    return acc
#
# def qdaTest(means,covmats,Xtest,ytest):
#     # Inputs
#     # means, covmats - parameters of the QDA model
#     # Xtest - a N x d matrix with each row corresponding to a test example
#     # ytest - a N x 1 column vector indicating the labels for each test example
#     # Outputs
#     # acc - A scalar accuracy value
#
#     # IMPLEMENT THIS METHOD
#     return acc
#
def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    w = inv((X).T.dot(X)).dot((X).T).dot(y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    N =  y.size
    I = np.eye(X.shape[1])
    w = inv((X).T.dot(X)+lambd*N*I).dot((X).T).dot(y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    Z = ytest-(Xtest.dot(w))
    Z = Z.T.dot(Z)
    N =  ytest.size
    rmse = sqrt( Z[0][0]) / N

    return rmse

# def regressionObjVal(w, X, y, lambd):
#
#     # compute squared error (scalar) and gradient of squared error with respect
#     # to w (vector) for the given data X and y and the regularization parameter
#     # lambda
#
#     # IMPLEMENT THIS METHOD
#     return error, error_grad
#

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    N = X.shape[0]
    w = np.mat(w).T

    error = (((y - X.dot(w)).T).dot((y - X.dot(w))) / (2*N)) + ((lambd * ((w.T).dot(w))) / 2)
    error_grad = (((((w.T).dot((X.T).dot(X))) - ((y.T).dot(X))) / N) + ((w.T) * lambd)).T
    error_grad = np.ndarray.flatten(np.array(error_grad))
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    x = x.reshape(-1,1)
    # print x.size
    tmp = np.ones((x.size,1))
    for i in range(1,p):
        tmp = np.hstack((tmp,np.power(x,i)))

    return tmp

# Main script
#
# Problem 1
# load the sample data
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
#
# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3tr = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3tr[i] = testOLERegression(w_l,X_i,y)
    i = i + 1

plt.plot(lambdas,rmses3)
plt.plot(lambdas,rmses3tr)
plt.show()

# # Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4train[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.plot(lambdas,rmses4train)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
rmses5train = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    rmses5train[p,0] = testOLERegression(w_d1,Xd,y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmses5train[p,1] = testOLERegression(w_d2,Xd,y)
plt.plot(range(pmax),rmses5)
plt.plot(range(pmax),rmses5train)
plt.legend(('No Regularization','Regularization',"No Regularization Train Data","Regularization Train Data"))
plt.show()