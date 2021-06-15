"""КР, Ларькин Владимир, М8О-303Б-18"""

import time
import numpy as np
import fire  # CLI
import multiprocessing as mp

from typing import List
from tqdm import tqdm  # прогресс-бары

import matplotlib.pyplot as plt

from utilities import parse_matrix  # парсинг матрицы из файла
from lab1_1.gauss import lu_solve


def lu_decomposition(matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """LU-разложение матрицы с выбором главного элемента.

    Так как в процессе разложения в матрице переставляются строки,
    дополнительно возвращается матрица перестановок P.

    :param matrix: входная матрица
    :return: кортеж из матриц P, L, U
    """

    # матрицы обязаны быть квадратными массивами размерности 2
    assert matrix.shape[0] == matrix.shape[1] and len(matrix.shape) == 2

    n = matrix.shape[0]

    l = np.zeros_like(matrix)
    u = np.copy(matrix)

    p = np.identity(n)

    for j in range(n - 1):
        m = np.abs(u[j:, j]).argmax() + j
        p[[j, m]] = p[[m, j]]
        l[[j, m]] = l[[m, j]]
        u[[j, m]] = u[[m, j]]

        for i in range(j + 1, n):
            l[i, j] = u[i, j] / u[j, j]

    for j in range(n - 1):
        for i in range(j + 1, n):
            for k in range(u.shape[1]):
                u[i, k] -= u[j, k] * l[i, j]

    l[np.diag_indices(n)] = 1
    return p, l, u


def split(a: np.ndarray, num: int) -> List[np.ndarray]:
    """Разделяет NumPy массив a на num - 1 равных частей + 1 часть из того, что осталось"""

    if num > len(a):
        return np.split(a, len(a))

    bound = len(a) // num * num
    res = np.split(a[:bound], num)
    rem = len(a) - len(a) // num * num
    if rem > 0:
        res[-1] = np.concatenate([res[-1], a[-rem:]])
    return res


def divide(data):
    b, c = data
    return b / c


def subtract(data):
    a, b, c = data
    return a - b.reshape(-1, 1) @ c.reshape(1, -1)


def lu_decomposition_parallel(matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Параллельное LU-разложение матрицы с выбором главного элемента.

    Так как в процессе разложения в матрице переставляются строки,
    дополнительно возвращается матрица перестановок P.

    :param matrix: входная матрица
    :return: кортеж из матриц P, L, U
    """

    if matrix.shape[0] != matrix.shape[1] or len(matrix.shape) != 2:
        raise ValueError("Матрицы обязаны быть квадратными массивами размерности 2")

    n = matrix.shape[0]

    l = np.zeros_like(matrix)
    u = np.copy(matrix)

    p = np.identity(n)

    proc_count = mp.cpu_count()
    pool = mp.Pool(proc_count)

    for j in range(n - 1):
        m = np.abs(u[j:, j]).argmax() + j
        p[[j, m]] = p[[m, j]]
        l[[j, m]] = l[[m, j]]
        u[[j, m]] = u[[m, j]]

        # аналогично l[j + 1:, j] = u[j + 1:, j] / u[j, j], но параллельно
        b = split(u[j + 1:, j], proc_count)
        c = [u[j, j]] * len(b)
        data_in = zip(b, c)
        data_out = pool.map(divide, data_in)
        l[j + 1:, j] = np.hstack(data_out)

        # for i in range(j + 1, n):
        #     u[i, :] -= u[j, :] * l[i, j]

        # for i in range(j + 1, n):
        #     for k in range(n):
        #         u[i, k] -= u[j, k] * l[i, j]

        # аналогично^, но одним умножением
        # так оказалось быстрее, чем inplace
        u[j + 1:, :] = u[j + 1:, :] - l[j + 1:, j].reshape(-1, 1) @ u[j, :].reshape(1, -1)

        # попытка распараллелить вот это^, но результат хуже, чем у простого цикла
        # a = split(u[j + 1:, :], proc_count)
        # b = split(l[j + 1:, j], proc_count)
        # c = [u[j, :]] * len(a)
        # data_in = zip(a, b, c)
        # data_out = pool.map(subtract, data_in)
        # u[j + 1:, :] = np.vstack(data_out)

    l[np.diag_indices(n)] = 1
    pool.close()
    return p, l, u


def main(src=None, test=False, shape=50, it=500):
    """Решение СЛАУ методом Гаусса с применением LU-разложения

    :param src: путь к текстовому файлу с матрицей
    :param test: флаг, запускающий тестирование
    :param shape: размер матриц, генерирующихся при тестировании
    :param it: число повторений тестирования
    """

    np.random.seed(42)

    if src is not None:
        # чтение файла
        with open(src, "r") as file:
            s = file.readlines()

        matrix = parse_matrix(s)

        a = matrix[:, :-1]
        b = matrix[:, -1]

        print("A:", a, sep="\n")
        print("b:", b)

        p, l, u = lu_decomposition_parallel(a)

        print(f"PLU:\n{p.T @ l @ u}")
        print(f"PLU == A: {np.allclose(p.T @ l @ u, a)}")

        print(f"P:\n{p}\nL:\n{l}\nU:\n{u}")

        x = lu_solve(l, u, p @ b)
        print(f"Решение системы: {x}")

    # тесты на случайно сгенерированных матрицах
    if test:
        run_test(shape, it)


def run_test(shape: int, it: int):
    """Тестирование LU-разложения и решения СЛАУ с замером времени и сравнением с функциями из numpy и scipy.

    :param shape: размер матриц
    :param it: количество тестов
    """

    print("\nТест решения СЛАУ:")
    times_my = {}
    times_par = {}

    shapes = list(range(10, shape, 10))

    for shape in tqdm(shapes):
        times_my[shape] = []
        times_par[shape] = []
        for _ in tqdm(range(it)):

            a = np.random.rand(shape, shape) * 100
            b = np.random.rand(shape) * 100

            prev = time.time_ns()

            p, l, u = lu_decomposition(a)

            _ = lu_solve(l, u, p @ b)

            times_my[shape].append((time.time_ns() - prev) / 1e9)

            if not np.allclose(p.T @ l @ u, a):
                print("Обычная")
                print(a)
                print(l)
                print(u)
                break

            prev = time.time_ns()

            pp, lp, up = lu_decomposition_parallel(a)
            _ = lu_solve(lp, up, pp @ b)

            times_par[shape].append((time.time_ns() - prev) / 1e9)

            if not np.allclose(pp.T @ lp @ up, a):
                print("Параллельная")
                print("L:\n", l)
                print("Lp:\n", lp)
                print("A:\n", a)
                break

    means = {}
    for name, d in zip(["Параллельная", "Последовательная"], [times_par, times_my]):
        means[name] = []
        for key in d:
            means[name].append(np.average(d[key]))

    for key in means:
        plt.plot(shapes, means[key], label=key)

    plt.legend()
    plt.grid(True)
    plt.title("Зависимость времени выполнения от размерности матрицы")
    plt.xlabel("Размерность матрицы")
    plt.ylabel("Время в секундах")

    plt.savefig("plot.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
