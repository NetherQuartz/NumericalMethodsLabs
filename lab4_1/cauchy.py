"""ЛР 4.1, Ларькин Владимир, М8О-303Б-18"""

import fire  # CLI
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

from utilities import str2fun
from lab3_4.derivative import first_derivative, second_derivative


def analytical_solution(f, a, b, h):
    x = [i for i in np.arange(a, b + h, h)]
    y = [f(i) for i in x]
    return x, y


def g(x, y, z):
    return z


def euler_method(f, g, a, b, h, y0, y_der):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    y = [y0]
    z = y_der
    for i in range(n):
        z += h * f(x[i], y[i], z)
        y_i = y[i] + h * g(x[i], y[i], z)
        y.append(y_i)
    return x, y


def runge_kutta_method(f, g, a, b, h, y0, y_der):
    n = int(np.ceil((b - a) / h))
    x = [i for i in np.arange(a, b + h, h)]
    y = [y0]
    z = [y_der]
    for i in range(n):
        K1 = h * g(x[i], y[i], z[i])
        L1 = h * f(x[i], y[i], z[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, z[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y.append(y[i] + delta_y)
        z.append(z[i] + delta_z)
    return x, y, z


def adams_method(f, g, x, y, z, h):
    n = len(x)
    x = x[:4]
    y = y[:4]
    z = z[:4]
    for i in range(3, n - 1):
        z_i = z[i] + h * (55 * f(x[i], y[i], z[i]) -
                          59 * f(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * f(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * f(x[i - 3], y[i - 3], z[i - 3])) / 24
        z.append(z_i)
        y_i = y[i] + h * (55 * g(x[i], y[i], z[i]) -
                          59 * g(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * g(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * g(x[i - 3], y[i - 3], z[i - 3])) / 24
        y.append(y_i)
        x.append(x[i] + h)
    return x, y


def runge_romberg_method(res):
    k = res[0]['h'] / res[1]['h']
    err_euler = []
    for i in range(len(res[0]['Euler']['y'])):
        err_euler.append(abs(res[0]['Euler']['y'][i] - res[1]['Euler']['y'][i]) / (k ** 1 - 1))

    err_runge = []
    for i in range(len(res[0]['Runge']['y'])):
        err_runge.append(abs(res[0]['Runge']['y'][i] - res[1]['Runge']['y'][i]) / (k ** 4 - 1))

    err_adams = []
    for i in range(len(res[0]['Adams']['y'])):
        err_adams.append(abs(res[0]['Adams']['y'][i] - res[1]['Adams']['y'][i]) / (k ** 4 - 1))

    return {'Euler': err_euler, 'Runge': err_runge, 'Adams': err_adams}


def exact_error(res, exact):
    err_euler = []
    for i in range(len(res[0]['Euler']['y'])):
        err_euler.append(abs(res[0]['Euler']['y'][i] - exact[0][1][i]))

    err_runge = []
    for i in range(len(res[0]['Runge']['y'])):
        err_runge.append(abs(res[0]['Runge']['y'][i] - exact[0][1][i]))

    err_adams = []
    for i in range(len(res[0]['Adams']['y'])):
        err_adams.append(abs(res[0]['Adams']['y'][i] - exact[0][1][i]))

    return {'Euler': err_euler, 'Runge': err_runge, 'Adams': err_adams}


def draw_plot(res, exact, *h):
    n = len(res)
    plt.rcParams["figure.figsize"] = (5, 4 * n)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(res[i]["Euler"]["x"], res[i]["Euler"]["y"], "-*", label='Эйлер')
        plt.plot(res[i]["Runge"]["x"], res[i]["Runge"]["y"], "-*", label='Рунге-Кутта')
        plt.plot(res[i]["Adams"]["x"], res[i]["Adams"]["y"], "-*", label='Адамс')
        plt.plot(exact[i][0], exact[i][1], "-*", label='Аналитическое решение')

        plt.legend()
        plt.title(f'$h_{i+1} = {h[i]}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    plt.savefig("plot.jpg", dpi=300)
    plt.show()


def main():
    """Решение задачи Коши методами Эйлера, Рунге-Кутты и Адамса 4-го порядка"""

    init_dict = {
        "a": 0,
        "b": 1,
        "h": 0.1,
        "y0": 1,
        "y_der0": 1,
        "equation": "-4 * x * y_der - (4 * x^2 + 2) * y",
        "solution": "(1 + x) * exp(-(x^2))"
    }

    # init_dict = {
    #     "a": 1,
    #     "b": 2,
    #     "h": 0.1,
    #     "y0": 2,
    #     "y_der0": 1,
    #     "equation": "12 * y / (x**2)",
    #     "solution": "x^4+x^(-3)"
    # }

    a, b = init_dict["a"], init_dict["b"]
    h_list = [init_dict["h"], init_dict["h"] / 2]
    equation = str2fun(init_dict["equation"], variables="x,y,y_der")
    y0, y_der0 = init_dict["y0"], init_dict["y_der0"]
    solution = str2fun(init_dict["solution"])
    save_res = []
    exact = []
    for h in h_list:
        euler_x, euler_y = euler_method(equation, g, a, b, h, y0, y_der0)
        runge_x, runge_y, runge_z = runge_kutta_method(equation, g, a, b, h, y0, y_der0)
        adams_x, adams_y = adams_method(equation, g, runge_x, runge_y, runge_z, h)
        anal_x, anal_y = analytical_solution(solution, a, b, h)
        exact.append((anal_x, anal_y))

        save_res.append({
            "h": h,
            "Euler": {'x': euler_x, 'y': euler_y},
            "Runge": {'x': runge_x, 'y': runge_y},
            "Adams": {'x': adams_x, 'y': adams_y}
        })

    errors_rr = runge_romberg_method(save_res)
    errors_abs = exact_error(save_res, exact)

    for errors in [errors_rr, errors_abs]:
        for key in errors.keys():
            errors[key] = np.average(errors[key])

    for title, errors in zip(["Средняя погрешность по Рунге-Ромбергу", "Средняя абсолютная погрешность"],
                      [errors_rr, errors_abs]):
        print(f"{title}:")
        for key in errors.keys():
            print(f"\t{key}:\t{errors[key]:.10f}")

    draw_plot(save_res, exact, h_list[0], h_list[1])


if __name__ == "__main__":
    fire.Fire(main)
