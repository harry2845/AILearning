import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
print("[%f %f]" %(x, y))
print("%d: x = %f, y = %f, z = %f" %(i, x, y, z))

while (z > 1e-2):
    x = x - (eta * calcdJ_x(x))
    y = y - (eta * calcdJ_y(y))
    z= calcJ(x, y)
    i = i+1

    print("[%f %f]" %(x, y))
    print("%d: x = %f, y = %f, z = %f" %(i, x, y, z))
    x_result.append(x)
    y_result.append(y)
    z_result.append(z)

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = calcJ(X[i, j], Y[i, j])

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap='rainbow')
plt.plot(x_result, y_result, z_result, c='black')
plt.show()
