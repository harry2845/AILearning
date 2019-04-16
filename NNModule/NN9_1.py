import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Parameters import *
from Activators import *
from TwoLayerClassificationNet import *
from DataReader import *

def ShowData(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        # end if
    # end for
    plt.show()

def ShowResult(net, X, Y, title, wb1, wb2):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.ForwardCalculationBatch(TX, wb1, wb2)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

def ShowAreaResult(X,Y,wb1,wb2,net):
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            A1 = net.ForwardCalculationBatch2(x,wb1,wb2)
            r = A1["Output"]
            if r[0,0] > 0.5:
                plt.plot(x[0,0], x[1,0], 's', c='y')
            else:
                plt.plot(x[0,0], x[1,0], 's', c='k')
            # end if
        # end for
    # end for
    ShowData(X, Y)

def Test(net, wb1, wb2, x1, x2):
    print("number one: ", x1)
    print("number two: ", x2)
    a = net.ForwardCalculationBatch2(np.array([x1, x2]).reshape(2,1),wb1,wb2)
    y = 0 if x1 == x2 else 1
    print(y)
    a1 = a["Output"]
    print(a1)
    if (np.abs(a1 - y)) < 1e-2:
        print("True")

if __name__ == '__main__':
    dataReader = DataReader(None, None)
    dataReader.X = np.array([0, 0, 1, 1, 0, 1, 0, 1]).reshape(2, 4)
    dataReader.Y = np.array([0, 1, 1, 0]).reshape(1, 4)
    dataReader.num_example = dataReader.X.shape[1]
    dataReader.num_feature = dataReader.X.shape[0]
    dataReader.num_category = len(np.unique(dataReader.Y))
    ShowData(dataReader.X, dataReader.Y)

    n_input, n_hidden, n_output = 2, 2, 1
    eta, batch_size, max_epoch = 0.5, 4, 50000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, LossFunctionName.CrossEntropy2)

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    net = TwoLayerClassificationNet()
    wb1, wb2 = net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    Test(net, trace.wb1, trace.wb2, 0, 0)
    Test(net, trace.wb1, trace.wb2, 0, 1)
    Test(net, trace.wb1, trace.wb2, 1, 0)
    Test(net, trace.wb1, trace.wb2, 1, 1)

    #ShowResult(net, dataReader.X, dataReader.Y, title, trace.wb1, trace.wb2)
    ShowAreaResult(dataReader.X, dataReader.Y, trace.wb1, trace.wb2, net)
