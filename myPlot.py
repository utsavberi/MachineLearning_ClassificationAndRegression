__author__ = 'utsav'
import numpy as np
import matplotlib.pyplot as plt

def plot_classes(X,y):
    colors = "bgrcmykw"
    color_index = 0

    classes = np.unique(y)
    for i in classes:
        X_curr = X[np.where(y==i)[0],]
        plt.scatter(X_curr[:,0],X_curr[:,1],c=colors[color_index])
        color_index += 1 % len(colors)
        # print np.random.rand(3,1)



    # plt.scatter(xtest[:,0],xtest[:,1])
    # plt.show()