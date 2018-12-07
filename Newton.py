import sympy as sp


def diff_rows(func, val):
    return [func.diff(v) for v in val]


x = sp.MatrixSymbol('x', 2, 1)
x_n = sp.MatrixSymbol('x_n', 2, 1)
hessian = sp.MatrixSymbol('hessian', 2, 2)
d = sp.MatrixSymbol('d', 2, 1)
lamda = sp.symbols("lamda")

gamma = 100

f = -9 * x[0] - 10 * x[1] + \
    gamma * (-sp.log(100 - x[0] - x[1]) - sp.log(x[0]) - sp.log(x[1]) - sp.log(50 - x[0] + x[1]))  # function

# Gradient
f_grad = sp.Matrix([f.diff(var) for var in x])
# Hessian
f_hess = sp.Matrix([diff_rows(j, x) for j in f_grad])

dk = sp.MatMul(-1 * hessian * f_grad)

x00 = sp.Matrix([[8], [90]])
x01 = sp.Matrix([[1], [40]])
x02 = sp.Matrix([[15], [68.69]])
x03 = sp.Matrix([[10], [20]])

for x0 in [x00, x01, x02, x03]:
    k = 0
    while True:
        h = f_hess.subs({x: x0}).evalf()
        h = h.inv('LU')
        d_ = sp.Matrix(dk.subs({x: x0, hessian: h}))

        if k > 100 or d_ == 0:
            break

        if not (x0[0] - x0[1] < 50 and x0[0] > 0 and x0[1] > 0 and x0[0] + x0[1] < 100):
            print("constraint!")
            break
        print(x0)
        x_next = x0 + lamda * d_

        l0 = 0.5
        l = l0
        k_back = 0
        theta = f.subs({x: x_next})
        theta_hat = theta.subs({x: 0}) + lamda * 0.5 * theta.diff(lamda).subs({x: 0})
        while True:
            if k_back == 0:
                l0 = 0.5
                k_back += 1
            else:
                theta_val = theta.subs({lamda: 10}).evalf()

                if sp.im(theta_val) == 0 and theta_val <= theta_hat.subs({lamda: l0}).evalf() or k_back > 10:
                    l = l0
                    break
                else:
                    l0 = l0 * 0.5
                    k_back += 1

        x_next = x_next.subs({lamda: l}).evalf()

        print("f: {}, x: {}, d: {}, k: {}".format(f.subs({x: x0}).evalf(), x0, d_, k))

        if not (x_next[0] - x_next[1] < 50 and x_next[0] > 0 and x_next[1] > 0 and x_next[0] + x_next[1] < 100):
            print("constraint!")
            break
        else:
            x0 = x_next
            k += 1
