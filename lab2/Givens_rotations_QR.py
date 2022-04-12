from sys import argv
from numpy import transpose, matmul, eye, sqrt, array, zeros
from numpy.linalg import norm
from numpy.random import rand
from random import random
import numpy

numpy.set_printoptions(linewidth=500)
numpy.set_printoptions(suppress=True)

def QR(A):
    m = len(A)
    n = len(A[0])

    Q = eye(m)
    R = A

    Gs = []

    for i in range(n):
        for j in range(i + 1, m):
            x1 = R[i][i]
            x2 = R[j][i]

            if x2 == 0:
                continue

            c = x1 / sqrt(x1**2 + x2**2)
            s = -x2 / sqrt(x1**2 + x2**2)

            G = eye(m)
            G[i][i] = c
            G[j][j] = c
            G[j][i] = s
            G[i][j] = -s
            
            R = matmul(G, R)
            
            Gs.append(G)

    for G in Gs:
        Q = matmul(Q, transpose(G))

    return Q, R

if __name__ == "__main__":
    m = int(argv[1])
    n = int(argv[2])

    A  = zeros((m, n))

    for i in range(m):
        for j in range(n):
            A[i][j] = random() * 200 - 100

    QR(A)
    print(f'A =\n{A}\n')
    Q,R = QR(A)
    print(f'Q =\n{Q}\n')
    print(f'R =\n{R}\n')
    print(f'|QR - A| = {norm(matmul(Q, R) - A)}')