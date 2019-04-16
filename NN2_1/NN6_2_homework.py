import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from BaseClassification import *

x_data_name = "X2.dat"
y_data_name = "Y2.dat"

def ToBool(YData):
    num_example = YData.shape[1]
    Y = np.zeros((1, num_example))
    for i in range(num_example):
        if YData[0,i] == 0:     # 第一类的标签设为0
            Y[0,i] = -1
        elif YData[0,i] == 1:   # 第二类的标签设为1
            Y[0,i] = 1
        # end if
    # end for
    return Y

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def Tanh(x):
    tanh = 2 * Sigmoid(2 * x) - 1
    return tanh

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    A = Tanh(Z)
    return A

def BackPropagationBatch(batch_X, batch_Y, A):
    m = batch_X.shape[1]
    dZ = (A - batch_Y) * 2
    # dZ列相加，即一行内的所有元素相加
    dB = dZ.sum(axis=1, keepdims=True)/m
    dW = np.dot(dZ, batch_X.T)/m
    return dW, dB

def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    A = ForwardCalculationBatch(W,B,X)
    
    p1 = 1 - Y
    p2 = np.log((1 - A) / 2)
    p3 = np.log((1 + A) / 2)

    p4 = np.multiply(p1 ,p2)
    p5 = np.multiply((1+Y), p3)

    LOSS = np.sum(-(p4 + p5))  #binary classification
    loss = LOSS / m
    return loss

def Inference(W,B,X_norm,xt):
    xt_normalized = NormalizePredicateData(xt, X_norm)
    A = ForwardCalculationBatch(W,B,xt_normalized)
    return A, xt_normalized

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

    for i in range(xt.shape[1]):
        plt.plot(xt[0,i], xt[1,i], '^', c='b')

    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()

# 主程序
if __name__ == '__main__':
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    XData,YData = ReadData(x_data_name, y_data_name)
    X, X_norm = NormalizeData(XData)
    Y = ToBool(YData)
    W, B = train(method, X, Y, ForwardCalculationBatch, CheckLoss)
    print("W=",W)
    print("B=",B)
    xt = np.array([5,1,6,9,5,5]).reshape(2,3,order='F')
    result, xt_norm = Inference(W,B,X_norm,xt)
    print(result)
    print(np.around(result))
    ShowResult(X, YData, W, B, xt_norm)
