"""ЛР 2.2, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from sympy import sympify, lambdify, diff


def get_q(x, phi1, phi2):
    dphi1_dx1 = lambdify("x1,x2", diff(phi1, "x1"))
    dphi1_dx2 = lambdify("x1,x2", diff(phi1, "x2"))
    dphi2_dx1 = lambdify("x1,x2", diff(phi2, "x1"))
    dphi2_dx2 = lambdify("x1,x2", diff(phi2, "x2"))

    max_phi1 = abs(dphi1_dx1(*x)) + abs(dphi1_dx2(*x))
    max_phi2 = abs(dphi2_dx1(*x)) + abs(dphi2_dx2(*x))

    return max(max_phi1, max_phi2)


def A1(x, f1, f2):
    df1_dx2 = lambdify("x1,x2", diff(f1, "x2"))
    df2_dx2 = lambdify("x1,x2", diff(f2, "x2"))
    f1 = lambdify("x1,x2", f1)
    f2 = lambdify("x1,x2", f2)

    return np.array([[f1(*x), df1_dx2(*x)],
                     [f2(*x), df2_dx2(*x)]])


def A2(x, f1, f2):
    df1_dx1 = lambdify("x1,x2", diff(f1, "x1"))
    df2_dx1 = lambdify("x1,x2", diff(f2, "x1"))
    f1 = lambdify("x1,x2", f1)
    f2 = lambdify("x1,x2", f2)

    return np.array([[df1_dx1(*x), f1(*x)],
                     [df2_dx1(*x), f2(*x)]])


def jacobian(x, f1, f2):
    df1_dx1 = lambdify("x1,x2", diff(f1, "x1"))
    df2_dx1 = lambdify("x1,x2", diff(f2, "x1"))
    df1_dx2 = lambdify("x1,x2", diff(f1, "x2"))
    df2_dx2 = lambdify("x1,x2", diff(f2, "x2"))

    return np.array([[df1_dx1(*x), df1_dx2(*x)],
                     [df2_dx1(*x), df2_dx2(*x)]])


def iteration_method(init_dict, count_it=False):
    interval, eps = init_dict['interval'], init_dict['eps']

    inter_x1, inter_x2 = interval[0], interval[1]

    phi1 = sympify(init_dict["phi1"])
    phi2 = sympify(init_dict["phi2"])

    x_prev = np.array([inter_x1[1] - inter_x1[0],
                       inter_x2[1] - inter_x2[0]]) / 2

    q = get_q(x_prev, phi1, phi2)

    phi1 = lambdify("x1,x2", phi1)
    phi2 = lambdify("x1,x2", phi2)

    c = 0
    while True:
        c += 1
        x = [phi1(*x_prev), phi2(*x_prev)]

        finish_iter = max([abs(i - j) for i, j in zip(x, x_prev)]) * q / (1 - q)
        if finish_iter <= eps:
            break

        x_prev = x

    return x if not count_it else (x, c)


def newton_method(init_dict, count_it=False):
    interval, eps = init_dict['interval'], init_dict['eps']

    inter_x1, inter_x2 = interval

    x_prev = np.array([inter_x1[1] - inter_x1[0], inter_x2[1] - inter_x2[0]]) / 2

    f1 = sympify(init_dict["f1"])
    f2 = sympify(init_dict["f2"])

    c = 0
    while True:
        c += 1

        x = np.array([x_prev[0] - np.linalg.det(A1(x_prev, f1, f2)) / np.linalg.det(jacobian(x_prev, f1, f2)),
                      x_prev[1] - np.linalg.det(A2(x_prev, f1, f2)) / np.linalg.det(jacobian(x_prev, f1, f2))])

        finish_iter = max([abs(i - j) for i, j in zip(x, x_prev)])
        if finish_iter <= eps:
            break

        x_prev = x

    return x if not count_it else (x, c)


def main(eps=0.01, test=False):
    """Решение нелинейного уравнения методами простой итерации и Ньютона

    :param eps: точность вычисления
    :param test: флаг, запускающий тестирование
    """

    init_dict = {
        "f1": "x1 - cos(x2) - 1",
        "f2": "x2 - log(x1 + 1, 10) - 3",
        "phi1": "cos(x2) + 1",
        "phi2": "log(x1 + 1, 10) + 3",
        "interval": [(-0.5, 0.5), (2.5, 3.5)],
        "eps": eps
    }

    print("eps =", init_dict["eps"])

    print("Функция f1(x) =", init_dict["f1"])
    print("Функция f2(x) =", init_dict["f2"])
    print("Функция phi1(x) =", init_dict["phi1"])
    print("Функция phi2(x) =", init_dict["phi2"])
    x, it = newton_method(init_dict, count_it=True)
    print(f"\nРешение методом Ньютона:\t\t\t{x} за {it} ит.")
    x, it = iteration_method(init_dict, count_it=True)
    print(f"Решение методом простой итерации:\t{x} за {it} ит.")

    if test:
        powers = np.arange(0, 11, .5)  # порядки эпсилонов (eps = 10 ** -p)
        mi = []
        mn = []
        for p in powers:
            init_dict["eps"] = 10 ** -p
            mi.append(iteration_method(init_dict, count_it=True)[1])
            mn.append(newton_method(init_dict, count_it=True)[1])

        plt.plot(powers, mi, label="Метод простых итераций")
        plt.plot(powers, mn, label="Метод Ньютона")
        plt.legend()
        plt.title("Зависимость числа итераций от точности")
        plt.xlabel("Порядок точности ($\\epsilon = 10^{-порядок}}$)")
        plt.ylabel("Число итераций")
        plt.xticks(powers[::2])
        plt.yticks(range(0, max(mi[-1], mn[-1]), 1))
        plt.grid(True)
        plt.savefig("benchmark.jpg", dpi=300)
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
