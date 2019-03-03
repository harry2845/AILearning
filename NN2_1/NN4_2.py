import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ReadData():
    Xfile = Path('TemperatureControlXData.dat')
    Yfile = Path('TemperatureControlYData.dat')
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X,Y
    else:
        return None,None

def CalcW(X, Y, m):
    return (m * sum(X * Y) - (sum(X) * sum(Y))) / (m * sum(X ** 2) - (sum(X) ** 2))

def CalcB(X, Y, w, m):
    return sum(Y - (w * X)) / m

def CalcGradientDiscent(X, Y, m, eta):
    w = 0.0
    b = 0.0
    J = []
    I = []
    for i in range(m):
        x = X[i]
        y = Y[i]
        z = w * x + b
        dz = z - y
        db = dz
        dw = dz * x
        w = w - eta * dw
        b = b - eta * db
        j = (z - y) ** 2 / 2
        J.append(j)
        I.append(i)
    print(j)
    plt.plot(I, J)
    plt.show()
    return w, b

def CalcGradientDiscent2(X, Y, m, eta, iter_num):
    w = 0.0
    b = 0.0
    J = []
    I = []
    for i in range(iter_num):
        dw = sum((w * X + b - Y) * X) / m
        db = sum(w * X + b - Y) / m
        w = w - eta * dw
        b = b - eta * db
        j = sum((w * X + b - Y) ** 2) / 2 / m
        J.append(j)
        I.append(i)
    print(j)
    plt.plot(I, J)
    plt.show()
    return w, b

if __name__ == '__main__':
    X, Y = ReadData()
    m = X.shape[0]
    w = CalcW(X, Y, m)
    print("w = %f" %(w))
    b = CalcB(X, Y, w, m)
    print("b = %f" %(b))
    
    eta = 0.1
    w, b = CalcGradientDiscent(X, Y, m, eta)
    print("w = %f, b = %f" %(w, b))

    iter_num = 500
    w, b = CalcGradientDiscent2(X, Y, m, eta, iter_num)
    print("w = %f, b = %f" %(w, b))