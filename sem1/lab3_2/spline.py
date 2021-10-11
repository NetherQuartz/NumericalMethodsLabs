"""ЛР 3.2, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from sem1.lab1_2.tdma import tdma_solve


def f(a, b, c, d, x):
    return a + b * x + c * (x ** 2) + d * (x ** 3)


def get_a(f):
    return [0] + [f[i] for i in range(0, len(f) - 1)]


def get_b(f, h, c):
    b = [0]
    n = len(f) - 1
    for i in range(1, n):
        b.append((f[i] - f[i - 1]) / h[i] - 1/3 * h[i] * (c[i + 1] + 2 * c[i]))
    b.append((f[n] - f[n - 1]) / h[n] - 2/3 * h[n] * c[n])
    return b


def get_c(x, f, h):
    n = len(f)
    a = [h[i - 1] for i in range(3, n)]
    b = [2 * (h[i - 1] + h[i]) for i in range(2, n)]
    c = [h[i] for i in range(2, n - 1)]
    d = np.array([3 * ((f[i] - f[i - 1]) / h[i] - ((f[i - 1] - f[i - 2]) / h[i - 1])) for i in range(2, n)])
    x = tdma_solve(np.array([
        [b[0], c[0], 0],
        [a[0], b[1], c[1]],
        [0, a[1], b[2]]
    ]), d)
    res = [0, 0, *x]
    return res


def get_d(h, c):
    d = [0]
    n = len(c) - 1
    for i in range(1, n):
        d.append((c[i + 1] - c[i]) / (3 * h[i]))
    d.append(-c[n] / (3 * h[n]))
    return d


def find_interval(points, x):
    for i in range(0, len(points) - 1):
        if points[i] <= x <= points[i + 1]:
            return i


def spline_interpolation(points, values, x):
    h = [0] + [points[i] - points[i - 1] for i in range(1, len(points))]
    c = get_c(points, values, h)
    a = get_a(values)
    b = get_b(values, h, c)
    d = get_d(h, c)

    i = find_interval(points, x)
    res = f(a[i + 1], b[i + 1], c[i + 1], d[i + 1], x - points[i])
    return res, a, b, c, d


def draw_plot(points, vals, a, b, c, d, x_star, y_star):
    x, y = [], []
    n = len(points) - 1
    for i in range(n):
        x1 = np.linspace(points[i], points[i + 1], 10, endpoint=True)
        y1 = [f(a[i + 1], b[i + 1], c[i + 1], d[i + 1], j - points[i]) for j in x1]
        x.append(x1)
        y.append(y1)

    for i in range(n):
        plt.plot(x[i], y[i], color="red")

    plt.plot(points, vals, "o")
    plt.plot(x_star, y_star, "o", color="purple", label=f"$x^*=[{x_star:.1f}, {y_star:.3f}]$")
    plt.legend()
    # plt.axis("equal")
    plt.grid(True)
    plt.savefig("plot.jpg", dpi=300)
    plt.show()


def main():
    """Интерполяция функции кубическим сплайном"""

    init_dict = {
        "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "y": [0.13534, 0.36788, 1.0, 2.7183, 7.3891],
        "x*": -0.5
    }

    val, a, b, c, d = spline_interpolation(init_dict["x"], init_dict["y"], init_dict["x*"])
    draw_plot(init_dict["x"], init_dict["y"], a, b, c, d, init_dict["x*"], val)

    print(f"f(x* = {init_dict['x*']}) = {val}")

    print("\nКоэффициенты:")
    for title, values in zip(["a", "b", "c", "d"], [a, b, c, d]):
        print(f"{title}:", *[f"{el:10.5f}" for el in values])


if __name__ == "__main__":
    fire.Fire(main)

