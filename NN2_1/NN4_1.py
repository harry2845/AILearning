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

def CalcW1(X, Y, m):
    return sum(Y * (X - np.mean(X))) / (sum(X ** 2) - (sum(X) ** 2 / m))

def CalcW2(X, Y, m):
    return sum(X * (Y - np.mean(Y))) / (sum(X ** 2) - (np.mean(X) *sum(X)))

if __name__ == '__main__':
    X, Y = ReadData()
    plt.plot(X,Y,'.')
    plt.show()
    m = X.shape[0]
    w = CalcW(X, Y, m)
    print("w = %f" %(w))
    b = CalcB(X, Y, w, m)
    print("b = %f" %(b))

    w1 = CalcW1(X, Y, m)
    print("w1 = %f" %w1)

    w2 = CalcW2(X, Y, m)
    print("w2 = %f" %w2)