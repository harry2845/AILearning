import math
import numpy as np
import matplotlib.pyplot as plt

fault = 0.001

print("how to play: 1) input x, 2) calculate c, 3) input target number but not faraway from c")
x = float(input("input x as initial number(1.2, 10), you can try 1.3:"))
print(x)
a = x * x
b = math.log(a)
c = math.sqrt(b)
plt.plot(x, c, 'rx')
print(c)
y = float(input("input y as target number(0.5, 2), you can try 1.8:"))
print(y)

delta_c = c- y
print("forward...")
print("x=%f, a=%f, b=%f, c=%f" %(x, a, b, c))
while (abs(delta_c) > 0.001):
    print("backward...")
    delta_b = delta_c * 2 * math.sqrt(b)
    delta_a = delta_b * a
    delta_x = delta_a / 2 / x
    print("delta_c=%f, delta_b=%f, delta_a=%f, delta_x=%f\n" %(delta_c, delta_b, delta_a, delta_x))

    print("forward...")
    x = x - delta_x
    a = x * x
    b = math.log(a)
    c = math.sqrt(b)
    plt.plot(x, c, 'rx')
    print("x=%f, a=%f, b=%f, c=%f" %(x, a, b, c))

    delta_c = c - y

print("done!")

x = np.linspace(1.1, 10, 1000)
y = [math.sqrt(2 * math.log(i)) for i in x]
z = [1/i/(math.sqrt(2 * math.log(i))) for i in x]
plt.plot(x, y)
plt.plot(x, z)
plt.show()