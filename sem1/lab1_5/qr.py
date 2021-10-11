"""ЛР 1.5, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from utilities import parse_matrix  # парсинг матрицы из файла


def householder(a, sz, k):
    v = np.zeros(sz)
    v[k] = a[k] + np.sign(a[k]) * np.linalg.norm(a[k:])
    for i in range(k + 1, sz):
        v[i] = a[i]
    v = v[:, np.newaxis]
    H = np.eye(sz) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def get_QR(A):
    sz = len(A)
    Q = np.identity(sz)
    A_i = np.copy(A)

    for i in range(sz - 1):
        col = A_i[:, i]
        H = householder(col, len(A_i), i)
        Q = Q @ H
        A_i = H @ A_i

    return Q, A_i


def get_roots(A, i):
    sz = A.shape[0]
    a11 = A[i, i]
    a12 = A[i, i + 1] if i + 1 < sz else 0
    a21 = A[i + 1, i] if i + 1 < sz else 0
    a22 = A[i + 1, i + 1] if i + 1 < sz else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def finish_iter_for_complex(A, eps, i):
    Q, R = get_QR(A)
    A_next = R @ Q
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return True if abs(lambda1[0] - lambda2[0]) <= eps and abs(lambda1[1] - lambda2[1]) <= eps else False


def get_eigenvalue(A, eps, i):
    A_i = np.copy(A)
    while True:
        Q, R = get_QR(A_i)
        A_i = R @ Q
        a = np.copy(A_i)
        if np.linalg.norm(a[i + 1:, i]) <= eps:
            res = (a[i][i], False, A_i)
            break
        elif np.linalg.norm(a[i + 2:, i]) <= eps and finish_iter_for_complex(A_i, eps, i):
            res = (get_roots(A_i, i), True, A_i)
            break
    return res


def QR_method(A, eps):
    res = []
    i = 0
    A_i = np.copy(A)
    while i < A.shape[0]:
        eigenval = get_eigenvalue(A_i, eps, i)
        if eigenval[1]:
            res += [*eigenval[0]]
            i += 2
        else:
            res.append(eigenval[0])
            i += 1
        A_i = eigenval[2]
    return np.array(res), i


def main(src, test=False, eps=0.01):
    """Нахождение собственных значений матрицы методом QR-разложения

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
    print(f"\neps={eps}\n")

    tmp, count_iter = QR_method(matrix, eps)
    print("QR_method:\n", tmp)

    print("\nnp.linalg.eig:\n", np.linalg.eig(matrix)[0])


if __name__ == "__main__":
    fire.Fire(main)
