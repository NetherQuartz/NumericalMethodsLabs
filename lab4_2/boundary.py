"""ЛР 4.2, Ларькин Владимир, М8О-303Б-18"""

import fire  # CLI
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

from utilities import str2fun
from lab1_2.tdma import tdma_solve
from lab3_4.derivative import first_derivative, second_derivative
from lab4_1.cauchy import runge_kutta_method


def g(x, y, z):
    return z


def first_der(x, y, x0):
    i = 0
    while i < len(x) - 1 and x[i + 1] < x0:
        i += 1
    return (y[i + 1] - y[i]) / (x[i + 1] - x[i])


def get_n(n_prev, n, ans_prev, ans, b, delta, gamma, y1):
    x, y = ans_prev[0], ans_prev[1]
    y_der = first_der(x, y, b)
    phi_n_prev = delta * y[-1] + gamma * y_der - y1
    x, y = ans[0], ans[1]
    y_der = first_der(x, y, b)
    phi_n = delta * y[-1] + gamma * y_der - y1
    return n - (n - n_prev) / (phi_n - phi_n_prev) * phi_n


def check_finish(x, y, b, delta, gamma, y1, eps):
    y_der = first_der(x, y, b)
    return abs(delta * y[-1] + gamma * y_der - y1) > eps


def shooting_method(f, g, a, b, alpha, beta, delta, gamma, y0, y1, h, eps):
    n_prev, n = 1.0, 0.8
    y_der = (y0 - alpha * n_prev) / (beta + eps)
    ans_prev = runge_kutta_method(f, g, a, b, h, n_prev, y_der)[:2]

    y_der = (y0 - alpha * n) / (beta + eps)
    ans = runge_kutta_method(f, g, a, b, h, n, y_der)[:2]

    while check_finish(ans[0], ans[1], b, delta, gamma, y1, eps):
        n, n_prev = get_n(n_prev, n, ans_prev, ans, b, delta, gamma, y1), n
        ans_prev = ans
        y_der = (y0 - alpha * n) / beta
        ans = runge_kutta_method(f, g, a, b, h, n, y_der)[:2]

    return ans


def finite_difference_method(f, p, q, a, b, alpha, beta, delta, gamma, y0, y1, h):
    n = int((b - a) / h)
    x = np.array([i for i in np.arange(a, b + h, h)])
    a = [0] + [1 - p(x[i]) * h / 2 for i in range(0, n - 1)] + [-gamma]
    b = [alpha * h - beta] + [q(x[i]) * h ** 2 - 2 for i in range(0, n - 1)] + [delta * h + gamma]

    c = [beta] + [1 + p(x[i]) * h / 2 for i in range(0, n - 1)]
    d = np.array([y0 * h] + [f(x[i]) * h ** 2 for i in range(0, n - 1)] + [y1 * h])

    m = np.zeros((len(b), len(b)))
    rng = np.arange(len(b))
    m[rng[:-1], rng[:-1] + 1] = c
    m[rng, rng] = b
    m[rng, rng - 1] = a

    y = tdma_solve(m, d)
    return x, y


def runge_romberg_method(res):
    k = res[0]['h'] / res[1]['h']
    err_shooting = []
    for i in range(len(res[0]['Shooting']['y'])):
        err_shooting.append(abs(res[0]['Shooting']['y'][i] - res[1]['Shooting']['y'][i]) / (k ** 1 - 1))

    err_fd = []
    for i in range(len(res[0]['FD']['y'])):
        err_fd.append(abs(res[0]['FD']['y'][i] - res[1]['FD']['y'][i]) / (k ** 1 - 1))

    return {'Shooting': err_shooting, 'FD': err_fd}


def exact_error(res, exact):
    err_shooting = []
    for i in range(len(res[0]['Shooting']['y'])):
        err_shooting.append(abs(res[0]['Shooting']['y'][i] - exact[0][1][i]))

    err_fd = []
    for i in range(len(res[0]['FD']['y'])):
        err_fd.append(abs(res[0]['FD']['y'][i] - exact[0][1][i]))

    return {'Shooting': err_shooting, 'FD': err_fd}


def draw_plot(res, exact, *h):
    n = len(res)
    plt.rcParams["figure.figsize"] = (6, 4 * n)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(res[i]["Shooting"]["x"], res[i]["Shooting"]["y"], "-*", label="М. стрельбы")
        plt.plot(res[i]["FD"]["x"], res[i]["FD"]["y"], "-*", label="М. конечных разностей")
        plt.plot(exact[i][0], exact[i][1], "-*", label="Аналитическое решение")
        plt.legend()
        plt.title(f'$h_{i+1} = {h[i]}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    plt.show()


def main():
    """Решение краевой задачи методами стрельбы и конечных разностей"""

    init_dict = {
        "a": 0,
        "b": np.pi / 6,
        "alpha": 1,
        "beta": 0,
        "delta": 1,
        "gamma": 0,
        "y0": 0,
        "y1": -np.sqrt(3) / 3,
        "h": 0.1,
        "eps": 1e-5,
        "equation": "2 * (1 + tan(x)^2) * y",
        "solution": "-tan(x)",
        "p": "0",
        "q": "-2 * (1 + tan(x)^2)",
        "f": "0"
    }

    # init_dict = {
    #     "a": 1,
    #     "b": 3,
    #     "alpha": 0,
    #     "beta": 1,
    #     "delta": 1,
    #     "gamma": -1,
    #     "y0": 0,
    #     "y1": 31/9,
    #     "h": 0.1,
    #     "eps": 1e-5,
    #     "equation": "2 * (y-(x+1)*y_der)/(x*(2*x+1))",
    #     "solution": "x+1+1/x",
    #     "p": "2*(x+1)/(x*(2*x+1))",
    #     "q": "-2/(x*(2*x+1))",
    #     "f": "0"
    # }

    a, b, alpha, beta = init_dict['a'], init_dict['b'], init_dict['alpha'], init_dict['beta']
    delta, gamma, y0, y1 = init_dict['delta'], init_dict['gamma'], init_dict['y0'], init_dict['y1']
    st, eps = init_dict['h'], init_dict['eps']

    p = str2fun(init_dict["p"], variables="x")
    q = str2fun(init_dict["q"], variables="x")
    f = str2fun(init_dict["f"], variables="x")

    func = str2fun(init_dict["equation"], variables="x,y,y_der")
    exact_func = str2fun(init_dict["solution"], variables="x")

    save_res = []
    steps = [st, st / 2]
    for h in steps:
        res = shooting_method(func, g, a, b, alpha, beta, delta, gamma, y0, y1, h, eps)
        res2 = finite_difference_method(f, p, q, a, b, alpha, beta, delta, gamma, y0, y1, h)

        save_res.append({
            "h": h,
            "Shooting": {'x': res[0], 'y': res[1]},
            "FD": {'x': res2[0], 'y': res2[1]}
        })

    exact = []
    for h in steps:
        x_exact = [i for i in np.arange(a, b + h, h)]
        y_exact = [exact_func(i) for i in x_exact]
        exact.append((x_exact, y_exact))

    # errors = Runge_Romberg_method(save_res)
    # errors2 = exact_error(save_res, exact)

    draw_plot(save_res, exact, st, st / 2)


if __name__ == "__main__":
    fire.Fire(main)
