import numpy as np
import matplotlib.pyplot as plt

def calcJ(x, y):
    return x * x + np.power(np.sin(y), 2)

def calcdJ_x(x):
    return 2 * x

def calcdJ_y(y):
    return 2 * np.sin(y) * np.cos(y)

x = 3
eta = 0.1
y = 1
z = calcJ(x, y)

x_result = [x]
y_result = [y]
z_result = [z]

i = 0

while (z > 1e-2):
    print("[%f %f]" %(x, y))
    print("%d: x = %f, y = %f, z = %f" %(i, x, y, z))

    x = x - (eta * calcdJ_x(x))
    y = y - (eta * calcdJ_y(y))
    z= calcJ(x, y)
    i = i+1
    x_result.append(x)
    y_result.append(y)

# x = np.linspace(-1.2, 1.2)
# y = [calcJ(i) for i in x]
# plt.plot(x, y)
# plt.plot(x_result, y_result)
# plt.show()
