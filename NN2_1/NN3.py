import numpy as np
import matplotlib.pyplot as plt

def calcJ(x):
    return x * x

def calcdJ(x):
    return 2 * x

x = -1
eta = 0.3
y = calcJ(x)

x_result = [x]
y_result = [y]

while (y > 1e-3):
    x = x - (eta * calcdJ(x))
    y = calcJ(x)
    x_result.append(x)
    y_result.append(y)
    print("x = %f, y = %f" %(x, y))

x = np.linspace(-1.2, 1.2)
y = [calcJ(i) for i in x]
plt.plot(x, y)
plt.plot(x_result, y_result)
plt.show()