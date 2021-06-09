"""ЛР 1.4, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from utilities import parse_matrix  # парсинг матрицы из файла


def max_el_indices(matrix):
    i_max = 0
    j_max = 1
    a_max = matrix[i_max, j_max]
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if abs(matrix[i, j]) > a_max:
                a_max = abs(matrix[i, j])
                i_max, j_max = i, j
    return i_max, j_max


def calculate_phi(a_ii, a_ij, a_jj):
    return np.pi / 4 if a_ii == a_jj else 0.5 * np.arctan(2 * a_ij / (a_ii - a_jj))


def t(matrix: np.ndarray) -> float:
    """Корень из суммы квадратов недиагональных элементов"""
    return np.sqrt((np.triu(matrix, k=1) ** 2).sum() + (np.tril(matrix, k=-1) ** 2).sum())


def rotation_eig(matrix: np.ndarray, eps: float = .01, benchmark=False) -> (np.ndarray, np.ndarray):
    """Метод вращений Якоби для нахождения собственных значений и собственных векторов симметрической матрицы

    :param matrix: матрица
    :param eps: точность
    :param benchmark: возвращать ли количество итераций и погрешность
    :return: собственные значения и вектора, а если benchmark=True, то ещё количество итераций и погрешность
    """
    a_i = np.copy(matrix)
    n = a_i.shape[0]
    eigenvectors = np.identity(n)

    c = 0
    while True:
        u = np.identity(n)
        i, j = max_el_indices(a_i)
        phi = calculate_phi(a_i[i, i], a_i[i, j], a_i[j, j])
        u[i, j] = -np.sin(phi)
        u[j, i] = np.sin(phi)
        u[i, i] = u[j, j] = np.cos(phi)

        a_i = u.T @ a_i @ u

        eigenvectors = eigenvectors @ u

        if t(a_i) < eps:
            break

        c += 1

    eigenvalues = np.diag(a_i)

    return (eigenvalues, eigenvectors) if not benchmark else (eigenvalues, eigenvectors, c, t(a_i))


def main(src, test=False, eps=0.01):
    """Решение СЛАУ методом простых итераций и методом Зейделя

    :param src: путь к текстовому файлу с матрицей
    :param test: флаг, запускающий тестирование
    :param eps: точность вычисления
    """

    np.random.seed(42)

    # чтение файла
    with open(src, "r") as file:
        s = file.readlines()

    matrix = parse_matrix(s)

    print(f"Матрица:\n{matrix}")
    print(f"\neps={eps}")

    print("\nnumpy.linalg.eig:", *np.linalg.eig(matrix), sep="\n")
    print("\nrotation_eig:", *rotation_eig(matrix, eps), sep="\n")

    if test:
        powers = np.arange(0, 10, .1)  # порядки эпсилонов (eps = 10 ** -p)
        counts = []
        errors = []
        for p in powers:
            temp = rotation_eig(matrix, 10 ** -p, benchmark=True)
            counts.append(temp[2])
            errors.append(temp[3])

        plt.plot(counts, errors, "-*")
        plt.title("Зависимость погрешности вычислений от числа итераций")
        plt.xlabel("Число итераций")
        plt.ylabel("Погрешность")
        plt.grid(True)
        plt.savefig("benchmark.jpg", dpi=300)
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
