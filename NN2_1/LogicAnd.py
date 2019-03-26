# Written by Dong Shen (shdon)

import numpy as np
import matplotlib.pyplot as plt
import math
from BaseClassification import *

def ReadData():
    X = np.array([0, 0, 1, 1, 0, 1, 0, 1]).reshape(2, 4)
    Y = np.array([0, 0, 0, 1]).reshape(1, 4)
    return X, Y

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    A = Sigmoid(Z)
    return A

def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    A = ForwardCalculationBatch(W,B,X)
    
    p1 = 1 - Y
    p2 = np.log(1-A)
    p3 = np.log(A)

    p4 = np.multiply(p1 ,p2)
    p5 = np.multiply(Y, p3)

    LOSS = np.sum(-(p4 + p5))  #binary classification
    loss = LOSS / m
    return loss

def Inference(W,B,X):
    xt_normalized = NormalizePredicateData(xt, X_norm)
    A = ForwardCalculationBatch(W,B,xt_normalized)
    return A, xt_normalized

def Test(W, B, x1, x2):
    print("number one: ", x1)
    print("number two: ", x2)
    a = ForwardCalculationBatch(W, B, np.array([x1, x2]).reshape(2,1))
    print(a)
    y = x1 or x2
    if (np.abs(a - y)) < 1e-2:
        print("True")

def ShowData(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        # end if
    # end for
    plt.show()

def ShowResult(X,Y,W,B,xt):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        # end if
    # end for

    b12 = -B[0,0]/W[0,1]
    w12 = -W[0,0]/W[0,1]

    x = np.linspace(0,1,10)
    y = w12 * x + b12
    plt.plot(x,y)

    #for i in range(xt.shape[1]):
    #    plt.plot(xt[0,i], xt[1,i], '^', c='b')

    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()

# 主程序
if __name__ == '__main__':
    # SGD, MiniBatch, FullBatch
    method = "FullBatch"
    # read data
    X,Y = ReadData()
    print(X)
    print(Y)
    # X, X_norm = NormalizeData(XData)
    # Y = ToBool(YData)
    W, B = train(method, X, Y, ForwardCalculationBatch, CheckLoss)
    print("W=",W)
    print("B=",B)
    Test(W, B, 0, 0)
    Test(W, B, 0, 1)
    Test(W, B, 1, 0)
    Test(W, B, 1, 1)
    ShowData(X, Y)
    xt = np.array([0,0,0,1,1,0,1,1]).reshape(2,4,order='F')
    print(xt)
    result = ForwardCalculationBatch(W,B,X)
    print("result=",result)
    print(np.around(result))
    ShowResult(X,Y,W,B,X)

