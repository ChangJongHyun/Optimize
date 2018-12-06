import sympy as sp


def diff_rows(func, val):
    return [func.diff(v) for v in val]


x = sp.MatrixSymbol('x', 2, 1)
x_n = sp.MatrixSymbol('x_n', 2, 1)
hessian = sp.MatrixSymbol('hessian', 2, 2)
d = sp.MatrixSymbol('d', 2, 1)
lamda = sp.symbols("lamda")

gamma = 10

f = -9 * x[0] - 10 * x[1] + \
    gamma * (-sp.log(100 - x[0] - x[1]) - sp.log(x[0]) - sp.log(x[1]) - sp.log(50 - x[0] + x[1]))  # function

# Gradient
f_grad = sp.Matrix([f.diff(var) for var in x])
# Hessian
f_hess = sp.Matrix([diff_rows(j, x) for j in f_grad])

dk = sp.MatMul(-1 * hessian * f_grad)

lamda = 1

x0 = sp.Matrix([[8], [90]])
k = 0

while True:
    h = f_hess.subs({x: x0}).evalf()
    h = h.inv('LU')
    d_ = dk.subs({x: x0, hessian: h})

    x_next = x0 + 1 * d_

    print(f.subs({x: x0}).evalf(), x0, x_next)

    if not (x_next[0] - x_next[1] < 50 and x_next[0] > 0 and x_next[1] > 0 and x_next[0] + x_next[1] < 100):
        print("constraint!")
        break
    else:
        x0 = x_next
        k += 1

