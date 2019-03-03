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

if __name__ == '__main__':
    X, Y = ReadData()
    plt.plot(X,Y,'.')
    plt.show()
    m = X.shape[0]
    w = CalcW(X, Y, m)
    print("w = %f" %(w))
    b = CalcB(X, Y, w, m)
    print("b = %f" %(b))