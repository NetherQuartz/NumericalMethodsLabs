"""ЛР 3.5, Ларькин Владимир, М8О-303Б-18"""

import fire  # CLI
import numpy as np
from sympy import integrate

from utilities import str2fun


def get_points(x0, x, step):
    return [i for i in np.arange(x0, x + step, step)]


def get_values(points, f):
    return [f(x) for x in points]


def rectangle_method(x, h, f):
    return h * sum([f((x[i] + x[i + 1]) / 2) for i in range(len(x) - 1)])


def trapeze_method(y, h):
    n = len(y) - 1
    return h * (y[0] / 2 + sum([y[i] for i in range(1, n - 1)]) + y[n] / 2)


def simpson_method(y, h):
    n = len(y) - 1
    return h / 3 * (y[0] + sum([4 * y[i] for i in range(1, n, 2)]) +
                    sum([2 * y[i] for i in range(2, n - 1, 2)]) + y[n])


def runge_romberg_method(res, anal_val):
    k = res[0]['h'] / res[1]['h']
    err_rec = [abs(res[0]['Rectangle'] - res[1]['Rectangle']) / (k ** 2 - 1), abs(res[0]['Rectangle'] - anal_val) / (k ** 2 - 1)]
    err_trap = [abs(res[0]['Trapeze'] - res[1]['Trapeze']) / (k ** 2 - 1), abs(res[0]['Trapeze'] - anal_val) / (k ** 2 - 1)]
    err_sim = [abs(res[0]['Simpson'] - res[1]['Simpson']) / (k ** 4 - 1), abs(res[0]['Simpson'] - anal_val) / (k ** 4 - 1)]
    return {'Rectangle': err_rec, 'Trapeze': err_trap, 'Simpson': err_sim}


def main():
    """Численное интегрирование методами прямоугольников, трапеций и Симпсона"""

    init_dict = {
        "x": (-1, 1),
        "h": [0.5, 0.25],
        "f": "x / ((2 * x + 7) * (3 * x + 4))"
    }

    a, b = init_dict["x"]
    h_list = init_dict["h"]
    f_str = init_dict["f"]

    anal_val = float(integrate(f_str, ("x", a, b)))
    f = str2fun(f_str)

    save_res = []
    for h in h_list:
        points = get_points(a, b, h)
        values = get_values(points, f)

        res_rec = rectangle_method(points, h, f)
        res_trap = trapeze_method(values, h)
        res_sim = simpson_method(values, h)

        save_res.append({
            "h": h,
            "Rectangle": res_rec,
            "Trapeze": res_trap,
            "Simpson": res_sim
        })

    errors = runge_romberg_method(save_res, anal_val)

    print("Интеграл f(x) =", f_str, "от", a, "до", b)
    print("Аналитическое решение:", anal_val)
    print(f"-------------------------------------")
    print(f"|  h  | Прямоуг.|  Трап.  | Симпсон |")
    print(f"-------------------------------------")
    for res in save_res:
        h = res["h"]
        rec = res["Rectangle"]
        trap = res["Trapeze"]
        simp = res["Simpson"]
        print(f"|{h:5.2f}|{rec:9.6f}|{trap:9.6f}|{simp:9.6f}|")
    print(f"-------------------------------------")
    print(f"|   Погрешности (Рунге-Ромберг):    |")
    print(f"-------------------------------------")

    for i in range(len(errors["Rectangle"])):
        print(f"|{h_list[i]:5.2f}|{errors['Rectangle'][i]:9.6f}|{errors['Trapeze'][i]:9.6f}|{errors['Simpson'][i]:9.6f}|")
    print(f"-------------------------------------")


if __name__ == "__main__":
    fire.Fire(main)
