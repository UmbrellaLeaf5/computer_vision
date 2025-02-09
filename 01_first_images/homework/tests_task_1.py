import pytest
import numpy as np
import cv2

from task_1 import GetPositionStatus


@pytest.fixture
def image() -> np.ndarray:
    """
    Фикстура для загрузки изображения лабиринта.

    Returns:
        np.ndarray: изображение лабиринта в формате numpy array.
    """

    return cv2.imread('task_1/20 by 20 orthogonal maze.png')


def test_get_position_status(image: np.ndarray):
    """
    Тест для проверки функции GetPositionStatus.
    Проверяет, правильно ли функция определяет статус позиции (цвет и расстояние) в четырех углах лабиринта.

    Args:
        image (np.ndarray): изображение лабиринта, предоставленное фикстурой image().
    """

    h, w, _ = image.shape

    expected1 = {'up': (7, np.array([0, 0, 0])),
                 'down': (0, np.array([255, 255, 255])),
                 'right': (8, np.array([0, 0, 0])),
                 'left': (7, np.array([0, 0, 0]))}

    expected2 = {'up': (0, np.array([255, 255, 255])),
                 'down': (8, np.array([0, 0, 0])),
                 'right': (0, np.array([255, 255, 255])),
                 'left': (7, np.array([0, 0, 0]))}

    expected3 = {'up': (7, np.array([0, 0, 0])),
                 'down': (0, np.array([255, 255, 255])),
                 'right': (8, np.array([0, 0, 0])),
                 'left': (0, np.array([255, 255, 255]))}

    expected4 = {'up': (0, np.array([255, 255, 255])),
                 'down': (8, np.array([0, 0, 0])),
                 'right': (8, np.array([0, 0, 0])),
                 'left': (0, np.array([255, 255, 255]))}

    def CompareDicts(d1: dict, d2: dict) -> bool:
        """
        Функция для сравнения двух словарей, содержащих информацию о статусе позиции.

        Args:
            d1 (dict): первый словарь для сравнения.
            d2 (dict): второй словарь для сравнения.

        Returns:
            bool: `True`, если словари идентичны, `False` в противном случае.
        """

        if d1.keys() != d2.keys():
            return False

        for key in d1:
            val1 = d1[key]
            val2 = d2[key]

            if not isinstance(val1, tuple) or not isinstance(val2, tuple) or len(val1) != 2 or len(val2) != 2:
                return False

            if val1[0] != val2[0]:
                return False

            arr1 = val1[1]
            arr2 = val2[1]

            if not isinstance(arr1, np.ndarray):
                arr1 = np.array(arr1)
            if not isinstance(arr2, np.ndarray):
                arr2 = np.array(arr2)

            if not np.array_equal(arr1, arr2):
                return False

        return True

    assert CompareDicts(GetPositionStatus(image, (5, 5)), expected1)
    assert CompareDicts(GetPositionStatus(image, (5, h - 5)), expected2)
    assert CompareDicts(GetPositionStatus(image, (w - 5, 5)), expected3)
    assert CompareDicts(GetPositionStatus(image, (w - 5, h - 5)), expected4)
