"""Вспомогательные функции, используемые в нескольких работах"""

import re
from typing import Iterable
import numpy as np


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
        r = re.findall(r"([-+]?\d+(\.\d+)?)", line)
        matrix_list.append([float(i[0]) for i in r])

    # проверка, что все строки содержат одинаковое число элементов
    n = len(matrix_list[0])
    for line in matrix_list[1:]:
        if len(line) != n:
            raise Exception(f"Ошибка чтения матрицы: разное количество элементов в строках")

    # создание и возврат матрицы
    return np.array(matrix_list)
