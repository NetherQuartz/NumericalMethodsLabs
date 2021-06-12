import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from lab1_1.gauss import lu_decomposition, lu_solve


def f(coeffs, x):
    """Вычисление значения полинома с коэффициентами coeffs"""
    return sum([x ** i * c for i, c in enumerate(coeffs)])


def sum_squared_errors(f, y):
    """Сумма квадратов ошибок"""
    return sum((f_i - y_i) ** 2 for f_i, y_i in zip(f, y))


def lsm(x, y, n):
    """Подбор коэффициентов полинома степени n с помощью МНК"""

    N = len(x)
    mat = [[sum([x_j ** (i + j) for x_j in x]) for i in range(n + 1)] for j in range(n + 1)]
    mat[0][0] = N + 1
    b = [sum([x_j ** i * y_j for x_j, y_j in zip(x, y)]) for i in range(n + 1)]

    mat = np.array(mat)
    b = np.array(b)
    p, l, u = lu_decomposition(mat)
    b = b @ p
    coeffs = lu_solve(l, u, b)

    return coeffs


def main():
    """Аппроксимация таблично заданной функции многочленами 1-й и 2-й степеней с помощью МНК"""

    init_dict = {
        "x": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
        "y": [0.04979, 0.13534, 0.36788, 1.0, 2.7183, 7.3891]
    }

    x, y = init_dict["x"], init_dict["y"]

    xc = np.arange(min(x) - 1, max(x) + 1, .01)

    c1 = lsm(x, y, 1)
    y1 = f(c1, xc)

    c2 = lsm(x, y, 2)
    y2 = f(c2, xc)

    plt.plot(x, y, "o", label="Входные данные")
    plt.plot(xc, y1, label="Полином первой степени")
    plt.plot(xc, y2, label="Полином второй степени")

    plt.title("Аппроксимация МНК")
    plt.grid(True)
    plt.legend()

    plt.savefig("plot.jpg", dpi=300)
    plt.show()

    e1 = sum_squared_errors(lsm(x, y, 1), y)
    e2 = sum_squared_errors(lsm(x, y, 2), y)

    print("Сумма квадратов ошибок:")
    print("\tn = 1:", e1)
    print("\tn = 2:", e2)


if __name__ == "__main__":
    fire.Fire(main)
