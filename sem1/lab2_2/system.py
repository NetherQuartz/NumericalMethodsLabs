"""ЛР 2.2, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from sympy import sympify, lambdify, diff, Expr
from numpy.linalg import det

from typing import Iterable, Callable

from utilities import str2fun


def get_q(x, phi):
    """Максимум модуля суммы по строке"""
    return np.abs(jacobian(x, phi).sum(axis=1)).max()


def get_a(x: np.ndarray, jacobi: np.ndarray, f: Iterable[Callable]) -> np.ndarray:
    """Возвращает тензор ixjxi, состоящий из i якобианов, в каждом из которых столбец i заменён на столбец из f_j(x)"""

    f_col = np.array([f_i(*x) for f_i in f])
    a = np.stack([jacobi] * jacobi.shape[1])
    for i in range(a.shape[0]):
        a[i, :, i] = f_col
    return a


def jacobian(x: np.ndarray, f: Iterable[Expr]) -> np.ndarray:
    """Матрица Якоби для списка функций f, вычисленная в точке x"""

    # получаем список переменных
    var_str = get_variables(f)
    var_list = var_str.split(",")

    # матрица ixj производных df_j/dx_i
    partials = [[lambdify(var_str, diff(fun, var)) for var in var_list] for fun in f]

    # вычисляем производные в точке и возвращаем
    return np.array([[fun(*x) for fun in row] for row in partials])


def get_variables(functions: Iterable[Expr]) -> str:
    """Возвращает строку вида "x1,x2,…", содержащую все переменные из функций в списке functions"""

    var_sets = [fun.free_symbols for fun in functions]
    var_set = set()
    for vs in var_sets:
        var_set |= vs

    var_list = sorted(map(str, var_set))
    return ",".join(var_list)


def iteration_method(init_dict, count_it=False):
    intervals, eps = init_dict['intervals'], init_dict['eps']

    phi = [sympify(fun) for fun in init_dict["phi"]]
    var_str = get_variables(phi)

    x_prev = np.array([inter[1] - inter[0] for inter in intervals]) / 2
    q = get_q(x_prev, phi)

    phi = [lambdify(var_str, fun) for fun in phi]

    c = 0
    while True:
        c += 1
        x = np.array([fun(*x_prev) for fun in phi])

        finish_iter = max([abs(i - j) for i, j in zip(x, x_prev)]) * q / (1 - q)
        if finish_iter <= eps:
            break

        x_prev = x

    return x if not count_it else (x, c)


def newton_method(init_dict, count_it=False):
    intervals, eps = init_dict['intervals'], init_dict['eps']

    f_expr = [sympify(fun) for fun in init_dict["f"]]
    var_str = get_variables(f_expr)

    x_prev = np.array([inter[1] - inter[0] for inter in intervals]) / 2

    f_call = [lambdify(var_str, fun) for fun in f_expr]

    c = 0
    while True:
        c += 1

        jacobi = jacobian(x_prev, f_expr)
        a = get_a(x_prev, jacobi, f_call)

        x = x_prev - np.array([det(a[i]) / det(jacobi) for i in range(a.shape[0])])

        finish_iter = max([abs(i - j) for i, j in zip(x, x_prev)])
        if finish_iter <= eps:
            break

        x_prev = x

    return x if not count_it else (x, c)


def main(eps=0.01, plot=False, test=False):
    """Решение системы нелинейных уравнений методами простой итерации и Ньютона

    :param eps: точность вычисления
    :param plot: флаг, отвечающий за вывод графика решения
    :param test: флаг, запускающий тестирование
    """

    init_dict = {
        "f": ["x1 - cos(x2) - 1", "x2 - log(x1 + 1, 10) - 3"],
        "phi": ["cos(x2) + 1", "log(x1 + 1, 10) + 3"],
        "intervals": [(-0.5, 0.5), (2.5, 3.5)],
        "eps": eps
    }

    print("eps =", init_dict["eps"], end="\n\n")

    # вывод всех функций f(x) и phi(x)
    print("Функции f(x):", *[f"\tf{i+1}(x) = {e}" for i, e in enumerate(init_dict["f"])], sep="\n", end="\n\n")
    print("Функции phi(x):", *[f"\tphi{i+1}(x) = {e}" for i, e in enumerate(init_dict["phi"])], sep="\n")

    # получение и вывод решений
    ans, it = newton_method(init_dict, count_it=True)
    print(f"\nРешение методом Ньютона:\t\t\t{ans} за {it} ит.")
    ans, it = iteration_method(init_dict, count_it=True)
    print(f"Решение методом простой итерации:\t{ans} за {it} ит.")

    if plot:
        y = np.arange(-10, 10, .001)
        phi1 = str2fun(init_dict["phi"][0])
        x = phi1(y)
        plt.plot(x, y, label=f"f1(x) = {init_dict['f'][0]}")

        x = np.arange(-0.999, 10, .001)
        phi2 = str2fun(init_dict["phi"][1])
        y = phi2(x)
        plt.plot(x, y, label=f"f2(x) = {init_dict['f'][1]}")

        plt.plot(*ans, "ro", label="Решение системы")

        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.xlim(init_dict["intervals"][0][0] - 2, init_dict["intervals"][0][1] + 2)
        plt.ylim(init_dict["intervals"][1][0] - 2, init_dict["intervals"][1][1] + 2)
        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.savefig("plot.jpg", dpi=300)
        plt.show()

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
