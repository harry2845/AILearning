import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def TargetFunction(x):
    return 3 * x + 1

def CreateSampleData(n):
    x = np.linspace(0, 1, n)
    noise = np.random.uniform(-0.5, 0.5, size=(n))
    print(noise)
    y = TargetFunction(x) + noise
    return x,y

def CostFuntion(x, y, a, count):
    c = (a - y) ** 2
    loss = c.sum() / count / 2
    return loss

def ShowResult(ax, x, y, a, loss, title):
    ax.scatter(x, y)
    ax.plot(x, a, 'r')
    titles = str.format("{0} Loss={1:01f}", title, loss)
    ax.set_title(titles)

n = 12
x, y = CreateSampleData(n)
plt.scatter(x, y)
plt.axis([0, 1.1, 0, 4.2])
plt.show()

ax = [0] * 4
b = 0
fig, ((ax[0], ax[1]), (ax[2], ax[3])) = plt.subplots(2, 2)
for i in range(0, 4):
    a = 3 * x + b
    loss = CostFuntion(x, y, a, n)
    ShowResult(ax[i], x, y, a, loss, str.format("3x+{0}", b))
    b = b + 0.5
plt.show()

b = 0
plt.scatter(x, y)
plt.axis([0, 1.1, 0, 4.2])
for i in range(0, 4):
    a = 3 * x + b
    loss = CostFuntion(x, y, a, n)
    plt.plot(x, a)
    b = b + 0.5
plt.show()

B = np.arange(0, 2, 0.05)
LOSS = []
for i in range(len(B)):
    a = 3 * x + B[i]
    loss = CostFuntion(x, y, a, n)
    LOSS.append(loss)
plt.plot(B, LOSS, 'x')
plt.show()

W = np.arange(2, 4, 0.05)
LOSS = []
for i in range(len(W)):
    a = W[i] * x + 1
    loss = CostFuntion(x, y, a, n)
    LOSS.append(loss)
plt.plot(W, LOSS, 'o')
plt.show()

def showWB():
    B = np.arange(-9, 11, 0.1)
    W = np.arange(-7, 13, 0.1)
    LOSS = np.zeros((len(W), len(B)))
    for i in range(len(W)):
        for j in range(len(B)):
            a = W[i] * x + B[j]
            loss = CostFuntion(x, y, a, n)
            LOSS[i, j] = loss
            print("LOSS[%d, %d] = %f" %(i, j, loss))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(W, B, LOSS, cmap='rainbow')
    plt.show()

W = np.linspace(2, 4, 100)
B = np.linspace(0, 2, 100)
X, Y = np.meshgrid(W, B)
Z = np.zeros((len(W), len(B)))
for i in range(len(W)):
    for j in range(len(B)):
        a = W[i] * x + B[j]
        Z[i, j] = CostFuntion(x, y, a, n)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap='rainbow')
ax.contour(X, Y, Z, zdir = 'z', levels = 20, offset = 0)
plt.show()

def test_2d(x,y,n):
    s = 200
    W = np.linspace(1,5,s)
    B = np.linspace(-2,3,s)
    LOSS = np.zeros((s,s))
    for i in range(len(W)):
        for j in range(len(B)):
            w = W[i]
            b = B[j]
            a = w * x + b
            loss = CostFuntion(x,y,a,n)
            LOSS[i,j] = round(loss, 2)
    print(LOSS)
    print("please wait for 20 seconds...")
    while(True):
        X = []
        Y = []
        is_first = True
        loss = 0
        for i in range(len(W)):
            for j in range(len(B)):
                if LOSS[i,j] != 0:
                    if is_first:
                        loss = LOSS[i,j]
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i,j] = 0
                        is_first = False
                    elif (LOSS[i,j] == loss):
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i,j] = 0
        if is_first == True:
            break
        plt.plot(X,Y,'.')
    
    plt.xlabel("w")
    plt.ylabel("b")
    plt.show()

test_2d(x, y, n)