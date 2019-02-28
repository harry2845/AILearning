w = 3
b = 4

def calcX(w, b):
    return 2 * w + 3 * b

def calcY(b):
    return 2 * b + 1

def calcZ(w, b):
    return calcX(w, b) * calcY(b)

fault = 1e-5
targetZ = 150
z = calcZ(w, b)
deltaZ = z - targetZ
print("double variable: w, b ------")
while (abs(deltaZ) > fault):
    print("w=%lf,b=%lf,z=%lf,delta_z=%lf" %(w, b, z, deltaZ))
    factorB = 3 * calcY(b) + 2 * calcX(w, b)
    factorW = 2 * calcY(b)
    deltaB = deltaZ / 2 / factorB
    deltaW = deltaZ / 2 / factorW
    print("factor_b=%lf,factor_w=%lf,delta_b=%lf,delta_w=%lf" %(factorB, factorW, deltaB, deltaW))
    b = b - deltaB
    w = w - deltaW
    z = calcZ(w, b)
    deltaZ = z - targetZ

print("done!")
print("final b=%f" %(b))
print("final w=%f" %(w))

