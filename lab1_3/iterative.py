"""ЛР 1.3, Ларькин Владимир, М8О-303Б-18"""

import numpy as np
import fire  # CLI
import matplotlib.pyplot as plt

from utilities import parse_matrix  # парсинг матрицы из файла
from lab1_1.gauss import lu_det, lu_decomposition, lu_inv


def diag_dominance(a: np.ndarray) -> bool:
    """Проверка матрицы на диагональное преобладание"""
    assert a.shape[0] == a.shape[1] and len(a.shape) == 2
    for i in range(a.shape[0]):
        s = sum(np.abs(a[i, :])) - 2 * np.abs(a[i, i])
        if s >= 0:
            return False
    return True


def matrix_norm(a: np.ndarray) -> float:
    """C-норма матрицы"""
    return max([sum(np.abs(row)) for row in a])


def vector_norm(v: np.ndarray) -> float:
    """C-норма вектора"""
    return max(np.abs(v))


def make_alpha_beta(a: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    """Матрица alpha и вектор beta для метода Якоби"""

    alpha = np.copy(a)
    beta = np.copy(b)

    n = a.shape[0]

    for j in range(n):
        m = np.abs(alpha[j:, j]).argmax() + j
        alpha[[j, m]] = alpha[[m, j]]
        beta[[j, m]] = beta[[m, j]]

    diag = np.copy(np.diag(alpha))

    for j in range(n):
        alpha[j, :] /= -diag[j]
        alpha[j, j] = 0
        beta[j] /= diag[j]

    return alpha, beta


def fixed_point(a: np.ndarray, b: np.ndarray, eps: float = 0.01, count_it=False) -> np.ndarray:
    """Метод простой итерации

    :param a: матрица коэффициентов
    :param b: вектор правых частей
    :param eps: точность
    :param count_it: если True, то будет возвращено количество проделанных итераций
    :return: решение системы, а если count_it=True, то кортеж из решения системы и числа итераций
    """

    assert a.shape[0] == a.shape[1] == b.shape[0] and len(a.shape) == 2
    assert lu_det(lu_decomposition(a)[2], drop_sign=True) != 0

    alpha, beta = make_alpha_beta(a, b)

    c = matrix_norm(alpha) / (1 - matrix_norm(alpha))
    x = np.copy(beta)
    eps_k = eps + 1
    count = 0
    while eps_k >= eps:
        x_new = beta + alpha @ x
        eps_k = c * vector_norm(x - x_new)
        x = x_new
        count += 1

    return x if not count_it else (x, count)


def seidel(a: np.ndarray, b: np.ndarray, eps: float = 0.01, count_it=False) -> np.ndarray:
    """Метод Зейделя

    :param a: матрица коэффициентов
    :param b: вектор правых частей
    :param eps: точность
    :param count_it: если True, то будет возвращено количество проделанных итераций
    :return: решение системы, а если count_it=True, то кортеж из решения системы и числа итераций
    """

    assert a.shape[0] == a.shape[1] == b.shape[0] and len(a.shape) == 2
    assert lu_det(lu_decomposition(a)[2], drop_sign=True) != 0

    alpha, beta = make_alpha_beta(a, b)

    B = np.tril(alpha)
    C = np.triu(alpha)

    B_inv = lu_inv(*lu_decomposition(np.identity(a.shape[0]) - B))
    m1 = B_inv @ C
    m2 = B_inv @ beta

    x = np.copy(beta)
    eps_k = eps + 1
    count = 1
    while eps_k >= eps:
        x_new = m1 @ x + m2
        eps_k = vector_norm(x - x_new)
        x = x_new
        count += 1

    return x if not count_it else (x, count)


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

    a = matrix[:, :-1]
    b = matrix[:, -1]

    print(f"A:\n{a}")
    print(f"b:\n{b}")

    print(f"eps={eps}")

    print(f"Метод простой итерации:\n{fixed_point(a, b, eps)}")
    print(f"Метод Зейделя:\n{seidel(a, b, eps)}")
    print(f"Реальное решение:\n{np.linalg.solve(a, b)}")

    # анализ влияния точности на число итераций
    if test:
        epsilons = np.arange(1e-3, 0.5, 5e-3)
        fp = []
        z = []
        for ep in epsilons:
            fp.append(fixed_point(a, b, ep, count_it=True)[1])
            z.append(seidel(a, b, ep, count_it=True)[1])

        plt.plot(epsilons, fp, label="Метод простых итераций")
        plt.plot(epsilons, z, label="Метод Зейделя")
        plt.legend()
        plt.title("Зависимость числа итераций от точности")
        plt.xlabel("$\epsilon$")
        plt.ylabel("Число итераций")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
