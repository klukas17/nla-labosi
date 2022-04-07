from sys import argv
from numpy import array, dot, transpose, matmul
from numpy.linalg import norm
from numpy.random import rand
import numpy

numpy.set_printoptions(linewidth=500)
numpy.set_printoptions(suppress=True)

def proj(u, a):
    return u * dot(u, a) / dot(u, u)

def QR(A):
    columns = []
    for i in range(min(len(A[0]), len(A))):
        columns.append(array([j[i] for j in A]))

    u = []
    for i in range(len(columns)):
        new_u = columns[i]
        for j in range(i):
            new_u -= proj(u[j], columns[i])
        u.append(new_u)

    e = []
    for vec in u:
        e.append(vec / norm(vec))

    Q = array([vec for vec in e])
    
    R = matmul(Q, A)
    Q = transpose(Q)

    return Q, R

m = int(argv[1])
n = int(argv[2])

A  = rand(m, n)
print(f'A =\n{A}\n')
Q,R = QR(A)
print(f'Q =\n{Q}\n')
print(f'R =\n{R}\n')
print(f'|QR - A| = {norm(matmul(Q, R) - A)}')