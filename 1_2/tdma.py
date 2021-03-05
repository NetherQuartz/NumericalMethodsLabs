import numpy as np
import fire
from utilities import parse_matrix


def main(src, test=False):
    """Решение СЛАУ с трёхдиагональной матрицей методом прогонки

    :param src: путь к текстовому файлу с матрицей
    :param test: флаг, запускающий тестирование
    """

    # чтение файла
    with open(src, "r") as file:
        s = file.readlines()

    matrix = parse_matrix(s)

    m = matrix[:, :-1]
    d = matrix[:, -1]

    b = np.diagonal(m)

    a = np.diagonal(m, offset=-1)
    a = np.hstack([0, a])

    c = np.diagonal(m, offset=1)
    c = np.hstack([c, 0])

    n = m.shape[0]  # размерность матрицы

    p = [-c[0] / b[0]]
    q = [d[0] / b[0]]
    for i in range(1, n):
        denominator = b[i] + a[i] * p[i - 1]
        p.append(-c[i] / denominator)
        q.append((d[i] - a[i] * q[i - 1]) / denominator)

    x = np.zeros(n)
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    print(f"Решение СЛАУ: {x}")

    if test:
        xt = np.linalg.solve(m, d)
        print("Тест пройден!" if np.allclose(x, xt) else f"Тест не пройден. Правильный ответ: {xt}")


if __name__ == "__main__":
    fire.Fire(main)
