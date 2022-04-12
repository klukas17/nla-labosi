from sys import argv
from numpy import array, linspace, matmul, transpose, vectorize, polyfit, polyval
from numpy.random import normal
from numpy.linalg import solve
from scipy.linalg import qr
from random import random
import matplotlib.pyplot as plt
import numpy
from Givens_rotations_QR import QR

numpy.set_printoptions(linewidth=500)
numpy.set_printoptions(suppress=True)

if len(argv) == 1:
    e = 0
else:
    e = int(argv[1])

f = lambda x : x**2 - 3*x + 2
m = vectorize(f)

x = []
y = []

for _ in range(50):
    x.append(random() * 6 - 3)

x.sort()

for i in range(50):
    y.append(f(x[i]))

x = array(x)
y = array(y)

A = []
b = []

for i in range(50):
    row = []
    for j in range(3):
        row.append(x[i]**j)
    A.append(row)
    b.append(y[i] + e * normal(0, 1))

A = array(A)
Q,R = QR(A)
d = matmul(transpose(Q), b)

d1 = array([d[i] for i in range(3)])

R = array([R[i] for i in range(3)])

c0, c1, c2 = solve(R, d1)

print(f'Procjena rješenja: {c2:.3f}x^2 + {c1:.3f}x + {c0:.3f}')
g = lambda x : c2 * x**2 + c1 * x + c0
n = vectorize(g)

xp = linspace(-3, 3, 60000)
u = m(xp)
v = n(xp)

a = polyfit(x, b, 2)
p = polyval(a, xp)

fig, ax = plt.subplots(figsize=(12, 12))
plt.plot(xp, u, 'r', label='Funkcija')
plt.plot(xp, p, 'b', label="Polyfit")
plt.plot(xp, v, 'g', label='Rješenje')
plt.plot(x, b, 'co', label="Podaci")
ax.legend(loc='upper right')
plt.show()