from numpy import array, matmul, zeros, dot, eye, transpose
from numpy.random import rand
from numpy.linalg import inv, norm, qr

from random import random

import numpy as np

D = array([[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]])
print(f'D = \n{D}')
T = rand(4, 4)
print(f'T = \n{T}')
A = matmul(matmul(T, D), inv(T))
print(f'A = T D T^-1 = \n{A}')

x = array([[random(), random(), random()], [random(), random(), random()], [random(), random(), random()], [random(), random(), random()]])
q,r = qr(x)
y = q[:,0:3]

iteration_count = 0
max_iterations = 10000

while iteration_count < max_iterations:
    x = matmul(A, y)
    q,r = qr(x)
    y = q[:,0:3]
    iteration_count += 1
    print(f'... {iteration_count}')

vectors = transpose(y)

for vec in vectors:
    print(norm(matmul(A, vec)) / norm(vec))