import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from BaseClassification import *

x_data_name = "X3.dat"
y_data_name = "Y3.dat"

def ToOneHot(YData, num_category):
    num_example = YData.shape[1]
    Y = np.zeros((num_category, num_example))
    for i in range(num_example):
        if YData[0,i] == 1:
            Y[0,i] = 1
        elif YData[0,i] == 2:
            Y[1,i] = 1
        elif YData[0,i] == 3:
            Y[2,i] = 1
        # end if
    # end for
    return Y

def Softmax(Z):
    shift_z = Z - np.max(Z, axis=0)
    exp_z = np.exp(shift_z)
    A = exp_z / np.sum(exp_z, axis=0)
    return A

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    A = Softmax(Z)
    return A

def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    A = ForwardCalculationBatch(W,B,X)
    p1 = np.log(A)
    p2 =  np.multiply(Y, p1)
    LOSS = np.sum(-p2) 
    loss = LOSS / m
    return loss

def Inference(W, B, x_norm, xt):
    xt_normalized = NormalizePredicateData(xt, X_norm)
    A = ForwardCalculationBatch(W,B,xt_normalized)
    r = np.argmax(A,axis=0)+1
    return A, xt_normalized, r

def ShowData(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 2:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        elif Y[0,i] == 3:
            plt.plot(X[0,i], X[1,i], '^', c='b')
        # end if
    # end for
    plt.show()

def ShowResult(X,Y,W,B,xt):
    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 2:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        elif Y[0,i] == 3:
            plt.plot(X[0,i], X[1,i], '^', c='b')
        # end if
    # end for

    # 公式一
    b12 = (B[1,0] - B[0,0])/(W[0,1] - W[1,1])
    w12 = (W[1,0] - W[0,0])/(W[0,1] - W[1,1])
    # 公式二
    b23 = (B[2,0] - B[1,0])/(W[1,1] - W[2,1])
    w23 = (W[2,0] - W[1,0])/(W[1,1] - W[2,1])
    # 公式三
    b13 = (B[2,0] - B[0,0])/(W[0,1] - W[2,1])
    w13 = (W[2,0] - W[0,0])/(W[0,1] - W[2,1])

    x = np.linspace(0,1,10)
    y = w12 * x + b12
    plt.plot(x,y)

    x = np.linspace(0,1,10)
    y = w23 * x + b23
    plt.plot(x,y)

    x = np.linspace(0,1,10)
    y = w13 * x + b13
    plt.plot(x,y)

    for i in range(xt.shape[1]):
        plt.plot(xt[0,i], xt[1,i], 'o', c='b')

    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()

def ShowLineResult(X,Y,W,B,xt):

    ShowAreaResult(X,Y,W,B)

    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 2:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        elif Y[0,i] == 3:
            plt.plot(X[0,i], X[1,i], '^', c='b')
        # end if
    # end for

    b12 = (B[1,0] - B[0,0])/(W[0,1] - W[1,1])
    w12 = (W[1,0] - W[0,0])/(W[0,1] - W[1,1])

    b23 = (B[2,0] - B[1,0])/(W[1,1] - W[2,1])
    w23 = (W[2,0] - W[1,0])/(W[1,1] - W[2,1])

    b13 = (B[2,0] - B[0,0])/(W[0,1] - W[2,1])
    w13 = (W[2,0] - W[0,0])/(W[0,1] - W[2,1])

    x = np.linspace(0,1,2)
    y = w13 * x + b13
    p13, = plt.plot(x,y)

    x = np.linspace(0,1,2)
    y = w23 * x + b23
    p23, = plt.plot(x,y)

    x = np.linspace(0,1,2)
    y = w12 * x + b12
    p12, = plt.plot(x,y)

    plt.legend([p13,p23,p12], ["13","23","12"])
    
    for i in range(xt.shape[1]):
        plt.plot(xt[0,i], xt[1,i], 'o')

    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()


def ShowAreaResult(X,Y,W,B):
    count = 100
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            A = ForwardCalculationBatch(W,B,x)
            r = np.argmax(A,axis=0)
            if r == 0:
                plt.plot(x[0,0], x[1,0], 's', c='y')
            elif r == 1:
                plt.plot(x[0,0], x[1,0], 's', c='k')
            elif r == 2:
                plt.plot(x[0,0], x[1,0], 's', c='g')
            # end if
        # end for
    # end for
    #plt.show()

# 主程序
if __name__ == '__main__':
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    XData,YData = ReadData(x_data_name, y_data_name)
    ShowData(XData, YData)
    X, X_norm = NormalizeData(XData)
    num_category = 3
    Y = ToOneHot(YData, num_category)
    W, B = train(method, X, Y, ForwardCalculationBatch, CheckLoss)

    print("W=",W)
    print("B=",B)
    xt = np.array([5,1,7,6,5,6,2,7]).reshape(2,4,order='F')
    a, xt_norm, r = Inference(W,B,X_norm,xt)
    print("Probility=", a)
    print("Result=",r)
    ShowLineResult(X,YData,W,B,xt_norm)
