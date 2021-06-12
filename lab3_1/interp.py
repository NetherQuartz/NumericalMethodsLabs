"""ЛР 3.1, Ларькин Владимир, М8О-303Б-18"""

import operator
from functools import reduce

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from utilities import str2fun


def omega(points, x, i):
    return reduce(operator.mul, [x - points[j] for j in range(len(points)) if i != j])


def get_coeffs(x, y):
    cnt = len(x)
    coefs = [y[i] for i in range(cnt)]

    for j in range(1, cnt):
        for i in range(cnt - 1, j - 1, -1):
            coefs[i] = float(coefs[i] - coefs[i - 1]) / float(x[i] - x[i - j])

    return coefs


def lagrange_interpolation(points, x, f):
    f = str2fun(f)
    res = 0
    for i in range(len(points)):
        f_w = f(points[i]) / omega(points, points[i], i)
        res += f_w * omega(points, x, i)

    return res


def newton_interpolation(points, x, f):
    f = str2fun(f)
    y = [f(i) for i in points]
    coefs = get_coeffs(points, y)
    cnt = len(coefs) - 1
    res = coefs[cnt]
    for i in range(cnt - 1, -1, -1):
        res = res * (x - points[i]) + coefs[i]
    return res


def main():
    """Интерполяция функции многочленами Лагранжа и Ньютона"""

    init_dict = {
        "f": "exp(x)",
        "x": [
            [-2, -1, 0, 1],
            [-2, -1, 0.2, 1]
        ],
        "x*": -0.5
    }

    x_input = init_dict["x"]
    f = init_dict["f"]
    x_star = init_dict["x*"]

    f_call = str2fun(f)

    plt.rcParams["figure.figsize"] = (5 * len(x_input), 4)

    for i, x_cur in enumerate(x_input):
        plt.subplot(1, len(x_input), i + 1)
        plt.title("$x \in$ " + str(x_cur))

        x = np.arange(-2, 1, .01)
        y = f_call(x)
        plt.plot(x, y, label=f"f(x) = {f}", linewidth=1.5)

        print("x =", x_cur)

        newton = []
        lagrange = []
        for x in x_cur:
            newton.append(newton_interpolation(x_cur, x, f))
            lagrange.append(lagrange_interpolation(x_cur, x, f))

        print("\tМногочлен Лагранжа:\t", np.array(lagrange))
        print("\tМногочлен Ньютона:\t", np.array(newton))
        y_star_lagrange = lagrange_interpolation(x_cur, x_star, f)
        y_star_newton = newton_interpolation(x_cur, x_star, f)
        y_true = f_call(x_star)
        print(f"\tПогрешность в точке x* = {x_star}")
        print("\t\tЛагранж:", abs(y_true - y_star_lagrange))
        print("\t\tНьютон: ", abs(y_true - y_star_newton))

        plt.plot(x_cur, newton, "-o", label="Ньютон", linewidth=3)
        plt.plot(x_cur, lagrange, ":*", label="Лагранж", linewidth=3)

        plt.legend()
        plt.axis("equal")
        plt.grid(True)
    plt.suptitle("Интерполяция функции многочленами Ньютона и Лагранжа")
    plt.savefig("plot.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)

