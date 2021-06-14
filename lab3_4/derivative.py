"""ЛР 3.4, Ларькин Владимир, М8О-303Б-18"""

import fire  # CLI


def der_several_brackets(x, X):
    n = len(X)
    if n == 1:
        return 1
    return brackets_mult(x, X[1:]) + brackets_mult(x, X[:1]) * der_several_brackets(x, X[1:])


def divdiff(X, Y):
    n = len(X)
    if n < 2:
        if n == 1:
            return Y[0]
        return None
    if n > 2:
        return (divdiff(X[1:], Y[1:]) - divdiff(X[:n-1], Y[:n-1])) / (X[n-1] - X[0])
    return (Y[1] - Y[0]) / (X[1] - X[0])


def brackets_mult(x, X):
    ans = 1
    for i in range(len(X)):
        ans *= (x - X[i])
    return ans


def brackets_mult1(x, X):
    ans = 0
    for i in range(len(X)):
        ans += brackets_mult(x, X[:i] + X[i+1:])
    return ans


def first_derivative(X, Y, x):
    """Первая производная таблично-заданной функции y(x)"""

    n = len(X)
    ans = 0
    for i in range(1, n):
        ans += divdiff(X[:i+1], Y[:i+1]) * brackets_mult1(x, X[:i])
    return ans


def brackets_mult2(x, X):
    ans = 0
    for i in range(len(X)):
        ans += der_several_brackets(x, X[:i] + X[i+1:])
    return ans


def second_derivative(X, Y, x):
    """Вторая производная таблично-заданной функции y(x)"""

    n = len(X)
    ans = 0
    for i in range(2, n):
        ans += divdiff(X[:i+1], Y[:i+1]) * brackets_mult2(x, X[:i])
    return ans


def main():
    """Нахождение производной таблично заданной функции y(x) в точке x*"""

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
