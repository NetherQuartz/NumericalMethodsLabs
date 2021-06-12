"""ЛР 3.4, Ларькин Владимир, М8О-303Б-18"""

import fire  # CLI


def find_interval(points, x):
    """Номер отрезка, в котором лежит точка x"""

    for i in range(0, len(points) - 1):
        if points[i] <= x <= points[i + 1]:
            return i


def first_derivative(x, y, x0):
    """Первая производная таблично-заданной функции y(x)"""

    i = find_interval(x, x0)
    addend1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    addend2 = ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - addend1) / \
              (x[i + 2] - x[i]) * (2 * x0 - x[i] - x[i + 1])
    return addend1 + addend2


def second_derivative(x, y, x0):
    """Вторая производная таблично-заданной функции y(x)"""

    i = find_interval(x, x0)
    num1 = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1])
    num2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return 2 * (num1 - num2) / (x[i + 2] - x[i])


def main():
    """Аппроксимация таблично заданной функции многочленами 1-й и 2-й степеней с помощью МНК"""

    init_dict = {
        "x": [-0.2, 0.0, 0.2, 0.4, 0.6],
        "y": [-0.20136, 0.0, 0.20136, 0.41152, 0.64350],
        "x*": 0.2
    }

    x, y = init_dict["x"], init_dict["y"]
    x_star = init_dict["x*"]

    first = first_derivative(x, y, x_star)
    second = second_derivative(x, y, x_star)

    print(f"x* = {x_star}")
    print(f"y'(x*) = {first:.4f}, y''(x*) = {second:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
