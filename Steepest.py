import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cmx

x = sp.MatrixSymbol('x', 2, 1)
d = sp.MatrixSymbol('d', 2, 1)
lamda = sp.symbols("lamda")

gamma = 10

# function
f = -9 * x[0] - 10 * x[1] + \
    gamma * (-sp.log(100 - x[0] - x[1]) - sp.log(x[0]) - sp.log(x[1]) - sp.log(50 - x[0] + x[1]))

# Gradient
f_grad = sp.Matrix([f.diff(var) for var in x])

# step size
theta = sp.MatMul(f_grad.transpose(), d)

# x_true = [[1], [40]]
# x_true = [[22.2816845159571], [69.6935491424287]]
x_true = [[10.5587703669977], [87.6109662895179]]

f_true = f.subs({x: sp.Matrix(x_true)}).evalf()

x00 = sp.Matrix([[8], [90]])
x01 = sp.Matrix([[1], [40]])
x02 = sp.Matrix([[15], [68.69]])
x03 = sp.Matrix([[10], [20]])

val = []

for i, x0 in enumerate([x00, x01, x02, x03]):
    x_ = x0
    f_val = []
    x_val = []
    # init
    k = 0

    while True:

        if not (x0[0] - x0[1] < 50 and x0[0] > 0 and x0[1] > 0 and x0[0] + x0[1] < 100):
            print("constraint!")
            break

        # find direction dk
        k_bi = 0  # bisection iter
        lk = 1  # lambda value
        ll = 0
        lu = lk

        dk = sp.Matrix(-1 * f_grad.subs({x: x0})).evalf()  # direction
        theta_k = sp.Matrix(theta.subs({x: x0 + lamda * dk, d: dk})).evalf()  # theta
        f_k = f.subs({x: x0}).evalf()
        f_val.append(f_k)
        x_val.append(x0)
        # terminate condition
        if dk == 0 or k > 30:
            break

        # bisection
        while True:
            # step 0 --> guess initial lambda
            if k_bi == 0:
                theta_ = theta_k.subs({lamda: lk}).evalf()[0]

                # -100 --> -90 으로 떨어짐
                if theta_ > 0 or theta_ < -20:
                    lu = lk
                    k_bi += 1
                else:
                    lk *= 2  # 2의 n제곱 으로 커짐
            #  stopping criteria
            elif k_bi is 20:
                break
            #  step k
            else:
                lk = (lu + ll) / 2
                theta_ = theta_k.subs({lamda: lk}).evalf()[0]

                if theta_ < 0:
                    ll = lk
                elif theta_ > 0:
                    lu = lk
                elif theta_ == 0:
                    break
                k_bi += 1

        x_next = x0 + lk * dk  # x k+1

        print("f: {}, x: {}, d: {} lambda: {}, k: {}".format(f.subs({x: x0}).evalf(), x0, dk, lk, k))

        if not (x_next[0] - x_next[1] < 50 and x_next[0] > 0 and x_next[1] > 0 and x_next[0] + x_next[1] < 100):
            print("constraint!")
            break
        else:
            x0 = x_next
            k += 1

    x_val = [sp.matrix2numpy(x) for x in x_val]
    x1 = [abs(x[0] - x_true[0])[0] for x in x_val]
    x2 = [abs(x[1] - x_true[1])[0] for x in x_val]
    x__ = [i + j for i, j in zip(x1, x2)]
    f_val = [abs(f_true - v) for v in f_val]
    print(f_true)
    # plot
    fig = plt.figure((i + 1), figsize=(10, 10))
    fig.suptitle("gamma = " + str(gamma))
    x_fig = fig.add_subplot(211)

    x_fig.set_title("||xk - x*|| - (" + str(x_[0]) + ", " + str(x_[1]) + ")")
    x_fig.set_xlabel("iter")
    x_fig.set_ylabel("x")
    x_fig.plot(range(k + 1), x__, marker='.', c='b')
    x_fig.set_xlim(0, 30)

    f_fig = fig.add_subplot(212)
    f_fig.set_title("||f(xk) - f(x*)||")
    f_fig.set_xlabel('iter')
    f_fig.set_ylabel('f')
    f_fig.plot(range(k + 1), f_val, marker='o', c='b')
    f_fig.set_xlim(0, 30)

plt.show()
