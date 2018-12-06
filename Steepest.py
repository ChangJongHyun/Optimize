import sympy as sp
import matplotlib.pyplot as plt

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
theta = sp.MatMul(f_grad.transpose().subs({x: x + lamda * d}), d)
# init
k = 0
x0 = sp.Matrix([[8], [90]])

dk = sp.Matrix(-1 * f_grad.subs({x: x0})).evalf()  # direction
theta_k = sp.Matrix(theta.subs({x: x0, d: dk}))[0]  # theta

val = []
for i in range(1, 200):
    print(theta_k.subs(lamda, i).evalf())
    val.append(theta_k.subs(lamda, i).evalf())

plt.plot(val)
plt.show()

while True:
    # find direction dk
    k_bi = 0  # bisection iter
    dk = sp.Matrix(-1 * f_grad.subs({x: x0})).evalf()  # direction
    theta_k = sp.Matrix(theta.subs({x: x0, d: dk}))[0]  # theta
    print(theta_k.subs(lamda, 50).evalf())

    # terminate condition
    if dk == 0:
        break

    # bisection
    # lambda value
    lk = 1
    ll = 0
    lu = lk
    while True:
        # step 0 --> guess initial lambda
        if k_bi == 0:
            theta_ = theta_k.subs({lamda: lk}).evalf()

            # -100 --> -90 으로 떨어짐
            if theta_ > 0 or theta_ > -100:
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
            theta_ = theta_k.subs({lamda: lk}).evalf()

            if theta_ < 0:
                ll = lk
            elif theta_ > 0:
                lu = lk
            elif theta_ == 0:
                break
            k_bi += 1

    print("f: {}, x: {}, d: {} lambda: {}, k: {}".format(f.subs({x: x0}).evalf(), x0, dk, lk, k))

    # constraint
    if not (x0[0] - x0[1] < 50 and x0[0] > 0 and x0[1] > 0 and x0[0] + x0[1] < 100):
        print("constraint!")
        break
    else:
        x0 = x0 + lk * dk  # x k+1
        k += 1