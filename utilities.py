"""Вспомогательные функции, используемые в нескольких работах"""

from typing import Iterable, List, Callable
import numpy as np
from sympy import sympify, lambdify, diff, Expr
import sympy


def parse_matrix(s: Iterable[str]) -> np.ndarray:
    """Принимает на вход коллекцию str, парсит её и возвращает матрицу.
    В случае неодинакового числа элементов в строках вызывает исключение.

    :param s: коллекция строк матрицы
    :return: разобранная матрица
    """

    # удаление пустых строк
    s = filter(lambda x: len(x) > 0 and x != "\n", s)

    # парсинг файла
    matrix_list = []
    for line in s:
        r = list(map(float, line.split()))
        matrix_list.append(r)

    # проверка, что все строки содержат одинаковое число элементов
    n = len(matrix_list[0])
    for line in matrix_list[1:]:
        if len(line) != n:
            raise Exception(f"Ошибка чтения матрицы: разное количество элементов в строках")

    # создание и возврат матрицы
    return np.array(matrix_list)


def diff_n_times(f: Expr, n: int) -> List[Expr]:
    """Принимает на вход выражение SymPy и возвращает список его первых n производных

    :param f: выражение, которое требуется продифференцировать
    :param n: число производных, которое требуется получить
    :return: список выражений SymPy, первое из которых — исходная функция, а остальные — её n производных
    """

    if n < 0:
        raise ValueError("Число производных не может быть отрицательным")

    derivatives = [f]
    for i in range(n):
        derivatives.append(diff(derivatives[i]))

    return derivatives


def str2fun(f: str, der_num=0, variables=None) -> Callable or List[Callable]:
    """Принимает на вход строку, представляющую собой выражение и возвращает её в виде функции или списка её производных

    :param f: строка — выражение, которое нужно преобразовать в функцию
    :param der_num: количество производных, если они нужны
    :param variables: строка вида "x1,x2,…", содержащая переменные — параметры выходной функции
    :return: функция или список из функции и её der_num производных
    """

    sympified = sympify(f)

    if len(sympified.free_symbols) > 1 and der_num > 0:
        raise ValueError("Для взятия производной функция должна содержать только один параметр")

    if variables is not None and der_num > 0 and variables != ",".join(map(str, sympified.free_symbols)):
        raise Exception("Взятие производной возможно только в функциях с одной переменной")

    if der_num > 0:
        derivatives = diff_n_times(sympified, der_num)
        return [lambdify(",".join(sorted(map(str, der.free_symbols))), der) for der in derivatives]
    else:
        variables = variables if variables is not None else ",".join(sorted(map(str, sympified.free_symbols)))
        return lambdify(variables, sympified)
