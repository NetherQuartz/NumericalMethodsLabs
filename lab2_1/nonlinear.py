"""ЛР 2.1, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt
from sympy import sympify, diff, lambdify

from utilities import str2fun


def get_q(a, b, dphi):
    return max(abs(dphi(a)), abs(dphi(b)))


def iteration_method(init_dict, count_it=False):
    interval, eps = init_dict['interval'], init_dict['eps']
    a, b = interval[0], interval[1]

    phi, dphi = str2fun(init_dict["phi"], der_num=1)

    x_prev = (b - a) / 2
    q = get_q(a, b, dphi)
    cnt_iter = 0

    while True:
        cnt_iter += 1

        x = phi(x_prev)

        finish_iter = abs(x - x_prev) * q / (1 - q)
        if finish_iter <= eps:
            break

        x_prev = x

    return x if not count_it else (x, cnt_iter)


def newton_method(init_dict, count_it=False):
    interval, eps = init_dict['interval'], init_dict['eps']
    a, b = interval[0], interval[1]

    x_prev = b
    cnt_iter = 0

    f, df = str2fun(init_dict["f"], der_num=1)

    while True:
        cnt_iter += 1

        x = x_prev - f(x_prev) / df(x_prev)

        finish_iter = abs(f(x) - f(x_prev))
        if finish_iter <= eps:
            break

        x_prev = x

    return x if not count_it else (x, cnt_iter)


def main(eps=0.01, test=False):
    """Решение нелинейного уравнения методами простой итерации и Ньютона

    :param eps: точность вычисления
    :param test: флаг, запускающий тестирование
    """

    init_dict = {
        "f": "exp(x) - 2 * x - 2",
        "phi": "log(2 + 2 * x)",
        "interval": (1, 2),
        "eps": eps
    }

    print("eps =", init_dict["eps"])

    print("Функция f(x) =", init_dict["f"])
    print("Функция phi(x) =", init_dict["phi"])
    x, it = newton_method(init_dict, count_it=True)
    print(f"\nРешение методом Ньютона:\t\t\t{x:.15f} за {it} ит.")
    x, it = iteration_method(init_dict, count_it=True)
    print(f"Решение методом простой итерации:\t{x:.15f} за {it} ит.")

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
        plt.yticks(range(0, max(mi[-1], mn[-1]), 5))
        plt.grid(True)
        plt.savefig("benchmark.jpg", dpi=300)
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
