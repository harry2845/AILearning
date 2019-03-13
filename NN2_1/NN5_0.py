import numpy as np
from pathlib import Path

def LoadData():
    Xfile = Path("HousePriceXData.dat")
    Yfile = Path("HousePriceYData.dat")
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

if __name__ == '__main__':
    X, Y = LoadData()
    print(X.shape)
    print(Y.shape)
    num_example = X.shape[1]
    one = np.ones((num_example, 1))
    print(one.shape)
    x = np.column_stack((one, (X[:, 0:num_example]).T))
    print(x.shape)
    y = Y.T
    print(y.shape)

    a = np.dot(x.T, x)
    print(a.shape)
    w = np.asmatrix(np.dot(x.T, x))
    print(w.shape)
    w = np.dot(np.dot(np.linalg.inv(a), x.T), y)
    print(w)
    b = w[0]
    w1 = w[1]
    w2 = w[2]
    w3 = w[3]
    price = b + 4 * w1 + w2 * 5 + 55 * w3
    print(price)
    print(price / 55)
