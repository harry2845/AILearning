import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ReadData():
    Xfile = Path('TemperatureControlXData.dat')
    Yfile = Path('TemperatureControlYData.dat')
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X.reshape(1, -1), Y.reshape(1, -1)
    else:
        return None,None

def ForwardCalculation(w, b, x):
    return w * x + b

def BackPropagation(x, y, z):
    dz = z - y
    db = dz
    dw = dz * x
    return dw, db

def UpdateWeights(w, b, dw, db, eta):
    w = w - eta * dw
    b = b - eta * db
    return w, b

def GetSample(X, Y, i):
    x = X[0, i]
    y = Y[0, i]
    return x, y

def ShowResult(X, Y, w, b, iteration):
    plt.plot(X, Y, 'b.')
    PX = np.linspace(0, 1, 10)
    PZ = w * PX + b
    plt.plot(PX, PZ, 'r')
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()
    print(iteration)
    print(w, b)

if __name__ == '__main__':
    X, Y = ReadData()

    m = X.shape[1]
    eta = 0.1
    w, b = 0, 0
    for i in range(m):
        x, y = GetSample(X, Y, i)
        z = ForwardCalculation(w, b, x)
        dw, db = BackPropagation(x, y, z)
        w, b = UpdateWeights(w, b, dw, db, eta)

    ShowResult(X, Y, w, b, 1)

    result = ForwardCalculation(w, b, 0.346)
    print("result = ", result)
