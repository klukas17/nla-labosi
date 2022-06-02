from numpy import array, matmul, zeros, dot, eye
from numpy.random import rand
from numpy.linalg import inv, norm

D = array([[10, 0, 0, 0], [0, 8, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]])
print(f'D = \n{D}')
T = rand(4, 4)
print(f'T = \n{T}')
A = matmul(matmul(T, D), inv(T))
print(f'A = T D T^-1 = \n{A}')

x = array([1, 1, 1, 1])
print(f'ITERATION 0: x = {x}')
y = x / norm(x)

iteration_count = 0

lambda_min = 0
lambda_max = 0

while True:
    x = matmul(A, y)

    Ax = matmul(A, x)

    if dot(Ax, x) == norm(Ax) * norm(x):
        print(f'RESULT = {norm(Ax) / norm(x)}')
        lambda_max = norm(Ax) / norm(x)
        break

    iteration_count += 1

    y = x / norm(x)
    print(f'ITERATION {iteration_count}: x = {x}')

B = inv(A)

x = array([1, 1, 1, 1])
print(f'ITERATION 0: x = {x}')
y = x / norm(x)

iteration_count = 0
while True:
    x = matmul(B, y)

    Bx = matmul(B, x)

    if dot(Bx, x) == norm(Bx) * norm(x):
        print(f'RESULT = {norm(x) / norm(Bx)}')
        lambda_min = norm(x) / norm(Bx)
        break

    iteration_count += 1

    y = x / norm(x)
    print(f'ITERATION {iteration_count}: x = {x}')

sigma = 5
C = A - (sigma * eye(4))
C = inv(C)

x = array([1, 1, 1, 1])
print(f'ITERATION 0: x = {x}')
y = x / norm(x)

lambda_sigma = 0
c = 0

iteration_count = 0
max_iterations = 1000
while iteration_count < max_iterations:
    x = matmul(C, y)

    Cx = matmul(C, x)

    if dot(Cx, x) == norm(Cx) * norm(x):
        print(f'RESULT = {norm(x) / norm(Cx) + sigma}')
        lambda_sigma = norm(x) / norm(Cx) + sigma
        break

    c = - norm(x) / norm(Cx)

    lambda_sigma = (c * sigma + 1) / c

    iteration_count += 1

    y = x / norm(x)
    print(f'ITERATION {iteration_count}: x = {x}')

print()
print(f'lambda_max = {lambda_max}')
print(f'lambda_min = {lambda_min}')
print(f'lambda_closest_to_sigma = {lambda_sigma}')