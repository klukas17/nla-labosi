from numpy import array, norm, matmul

def Jacobi_method(A, x, b, iteration_count, n):
    results = []
    residuals = []
    for _ in range(iteration_count):
        new_x = []
        for i in range(n):
            tmp = 0
            for j in range(n):
                if i != j:
                    tmp += A[i][j] * x[j]
            new_x.append((b[i] - tmp) / A[i][i])
        x = array(new_x)
        results.append(x)
        residuals.append(norm(matmul(A,x) - b))
    return results, residuals