from sys import argv
from numpy import array, eye, zeros, matmul
from numpy.linalg import norm
from numpy.random import rand
import numpy

numpy.set_printoptions(linewidth=500)
numpy.set_printoptions(suppress=True)

def sign(x):
    return 1 if x >= 0 else -1

def e(dim):
    ret_val = zeros(dim)
    ret_val[0] = 1
    return ret_val

def QR(A):

    m = len(A)
    n = len(A[0])
    
    Q = eye(m)
    R = A

    for i in range(n):
        u = array([j[i] for j in R])[i:]
        v = u + sign(u[0]) * norm(u) * e(m-i)
        v_t = v.reshape(-1, 1)
        H = eye(m-i) - 2 * (v * v_t) / matmul(v, v_t)
        H_ = eye(m)
        offset = i
        for j in range(len(H)):
            for k in range(len(H[0])):
                H_[j + offset][k + offset] = H[j][k]

        Q = matmul(Q, H_)
        R = matmul(H_, R)

    return Q, R

m = int(argv[1])
n = int(argv[2])

A  = rand(m, n)
print(f'A =\n{A}\n')
Q,R = QR(A)
print(f'Q =\n{Q}\n')
print(f'R =\n{R}\n')
print(f'|QR - A| = {norm(matmul(Q, R) - A)}')