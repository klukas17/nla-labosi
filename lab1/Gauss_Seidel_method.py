from numpy import  norm, matmul

def Gauss_Seidel_method(A, x, b, iteration_count, n):
    results = []
    residuals = []
    for _ in range(iteration_count):
        for i in range(n):
            tmp = 0
            for j in range(n):
                if i != j:
                    tmp += A[i][j] * x[j]
            x[j] = (b[i] - tmp) / A[i][i]
        results.append(x)
        residuals.append(norm(matmul(A,x) - b))
    return results, residuals