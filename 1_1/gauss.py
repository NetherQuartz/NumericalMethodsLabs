"""ЛР 1.1, Ларькин Владимир, М8О-303Б-18"""

import time
import numpy as np
import scipy.linalg
import fire  # CLI
from tqdm import tqdm  # прогресс-бары
from utilities import parse_matrix  # парсинг матрицы из файла


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
            u[i, :] -= u[j, :] * l[i, j]

    l[np.diag_indices(n)] = 1
    return p, l, u


def perm_parity(p: np.ndarray) -> int:
    """Вычисление чётности перестановки, заданной матрицей перестановки.

    :param p: матрица перестановки
    :return: 1, если перестановка чётная, и -1, если нечётная
    """

    # матрица обязана быть квадратным массивом размерности 2
    assert p.shape[0] == p.shape[1] and len(p.shape) == 2

    n = p.shape[0]  # размерность матрицы
    v = p @ np.arange(n)  # получаем массив индексов перестановки

    # ищем все инверсии в массиве, их число такой же чётности, что и перестановка
    parity = 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if v[i] > v[j]:
                parity *= -1

    return parity


def lu_det(u: np.ndarray, p: np.ndarray) -> float:
    """Вычисление определителя матрицы по её LU-разложению

    :param u: верхняя треугольная матрица LU-разложения
    :param p: матрица перестановок
    :return: определитель
    """

    return perm_parity(p) * np.product(np.diagonal(u))


def lu_solve(l: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение СЛАУ, прошедшей через LU-разложение.
    Требуется предварительно умножить вектор правых частей на матрицу перестановки.

    :param l: нижняя треугольная матрица
    :param u: верхняя треугольная матрица
    :param b: вектор правых частей СЛАУ
    :return: вектор-решение СЛАУ
    """

    n = l.shape[0]
    z = np.zeros_like(b)
    z[0] = b[0]
    for i in range(1, n):
        s = 0
        for j in range(i):
            s += l[i, j] * z[j]
        z[i] = b[i] - s

    x = np.zeros_like(b)
    x[-1] = z[-1] / u[-1, -1]
    for i in range(n - 2, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += u[i, j] * x[j]
        x[i] = (z[i] - s) / u[i, i]

    return x


def lu_inv(p: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Обращение матрицы с помощью LU-разложения

    :param p: матрица перестановок
    :param l: нижняя треугольная матрица
    :param u: верхняя треугольная матрица
    :return: обратная матрица
    """

    # матрица обязана быть невырожденной
    assert lu_det(u, p) != 0

    n = u.shape[0]
    inv = p @ np.identity(n)  # оптимизация памяти путём перезаписи столбцов единичной матрицы

    # решаем СЛАУ LUX=PE
    for j in range(n):
        inv[:, j] = lu_solve(l, u, inv[:, j])
    return inv


def main(src, test=False, shape=50, it=500):
    """Решение СЛАУ методом Гаусса с применением LU-разложения

    :param src: путь к текстовому файлу с матрицей
    :param test: флаг, запускающий тестирование
    :param shape: размер матриц, генерирующихся при тестировании
    :param it: число повторений тестирования
    """

    np.random.seed(42)

    # чтение файла
    with open(src, "r") as file:
        s = file.readlines()

    matrix = parse_matrix(s)

    a = matrix[:, :-1]
    b = matrix[:, -1]

    # a = np.random.rand(4, 4) * 1000

    print("A:", a, sep="\n")
    print("b:", b)

    p, l, u = lu_decomposition(a)

    print(f"PLU:\n{p.T @ l @ u}")

    print(f"P:\n{p}\nL:\n{l}\nU:\n{u}")

    x = lu_solve(l, u, p @ b)
    print(f"Решение системы: {x}")
    # print(np.linalg.solve(a, b))

    print(f"Определитель матрицы A: {lu_det(u, p)}")
    # print(np.linalg.det(a))

    inv = lu_inv(p, l, u)
    print("Обратная матрица:", inv, sep="\n")
    # print(np.linalg.inv(a))

    print(f"AA^-1=E: {np.allclose(np.identity(a.shape[0]), a @ inv)}")

    # тесты на случайно сгенерированных матрицах
    if test:
        run_test(shape, it)


def run_test(shape: int, it: int):
    """Тестирование LU-разложения, решения СЛАУ и обращения матриц с замером времени и сравнением
    с функциями из numpy и scipy.

    :param shape: размер матриц
    :param it: количество тестов
    """
    print(f"\nТест времени работы LU-разложения матриц {shape}x{shape}, {it} итераций:")

    times_my = []
    times_sp = []
    for _ in tqdm(range(it)):

        a = np.random.rand(shape, shape) * 100

        prev = time.time_ns()

        p, l, u = lu_decomposition(a)

        times_my.append(time.time_ns() - prev)

        prev = time.time_ns()

        scipy.linalg.lu(a)

        times_sp.append(time.time_ns() - prev)

        if not np.allclose(p.T @ l @ u, a):
            print(a)
            print(l)
            print(u)
            break

    print(f"\nВремя lu_decomposition:\t{np.average(times_my) * 1e-9:.10f} секунд")
    print(f"Время scipy.linalg.lu:\t{np.average(times_sp) * 1e-9:.10f} секунд")

    print("\nТест решения СЛАУ:")
    times_my = []
    times_np = []
    for i in tqdm(range(it)):
        a = np.random.rand(shape, shape) * 100
        p, l, u = lu_decomposition(a)
        b = np.random.rand(shape) * 100
        pb = p @ b

        prev = time.time_ns()

        x = lu_solve(l, u, pb)

        times_my.append(time.time_ns() - prev)

        prev = time.time_ns()

        z = np.linalg.solve(l, pb)
        xn = np.linalg.solve(u, z)

        times_np.append(time.time_ns() - prev)

        if not np.allclose(x, xn):
            times_my.pop(-1)
            times_np.pop(-1)
            print(a)
            print(b)
            break

    print(f"\nПройдено тестов {i + 1}/{it}")
    print(f"Время lu_solve: \t\t\t{np.average(times_my) * 1e-9:.10f} секунд")
    print(f"Время numpy.linalg.solve: \t{np.average(times_np) * 1e-9:.10f} секунд")

    print("\nТест обращения:")
    times_my = []
    times_np = []
    for i in tqdm(range(it)):
        a = np.random.rand(shape, shape) * 100

        prev = time.time_ns()

        inv = lu_inv(*lu_decomposition(a))

        times_my.append(time.time_ns() - prev)

        prev = time.time_ns()

        invn = np.linalg.inv(a)

        times_np.append(time.time_ns() - prev)

        if not np.allclose(inv, invn):
            times_my.pop(-1)
            times_np.pop(-1)
            print(a)
            print(b)
            break

    print(f"\nПройдено тестов {i + 1}/{it}")
    print(f"Время lu_inv: \t\t\t{np.average(times_my) * 1e-9:.10f} секунд")
    print(f"Время numpy.linalg.inv: {np.average(times_np) * 1e-9:.10f} секунд")


if __name__ == "__main__":
    fire.Fire(main)
