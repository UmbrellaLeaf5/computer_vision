import cv2
import numpy as np

Point = tuple[int, int]


def Rotate(image: np.ndarray, point: tuple, angle: float) -> np.ndarray:
    """
    Поворачивает изображение по часовой стрелке на угол от 0 до 360 градусов
    и меняет размер изображения.

    Args:
        image (np.ndarray): исходной изображение (матрица цветов) для поворота.
        point (tuple): координаты центра вращения.
        angle (float): угол поворота в градусах: от 0 до 360.

    Raises:
        ValueError: Rotate: angle should be between 0 and 360.

    Returns:
        np.ndarray: повернутое изображение с измененным размером.
    """

    if angle > 360 or angle < 0:
        raise ValueError("Rotate: angle should be between 0 and 360.")

    while angle > 90:
        image = Rotate(image, point, 90)
        angle -= 90

    height, width, _ = image.shape
    angle = np.deg2rad(angle)

    transition_matrix = np.matrix(
        [[np.sin(angle), np.cos(angle)],
         [np.cos(angle), np.sin(angle)]]
    )

    new_width, new_height = np.ceil(transition_matrix * np.array([[height], [width]]))
    new_width, new_height = int(new_width), int(new_height)

    src_pos = np.float32(np.array(
        [[0, 0],
         [0, height],
         [width, 0]]
    ))

    dst_pos = np.float32(np.array(
        [[0, width * np.sin(angle)],
         [height * np.sin(angle), new_height],
         [width * np.cos(angle), 0]]
    ))

    return cv2.warpAffine(image.copy(),
                          cv2.getAffineTransform(src_pos, dst_pos),  # type: ignore
                          (new_width, new_height))  # type: ignore


def FindCorners(mask: np.ndarray) -> list[Point]:
    """
    Находит четыре угла бинарной маски (тетради).

    Ищет первый ненулевой пиксель вдоль каждой границы маски.
    Углы возвращаются в следующем порядке:
    верхний левый, нижний левый, верхний правый, нижний правый.

    Args:
        mask (np.ndarray): бинарная маска.

    Returns:
        list[Point]: список из четырех кортежей (x, y), представляющих координаты углов.
    """

    corners: list[Point] = []

    for is_transposed in [False, True]:
        array = np.transpose(mask) if is_transposed \
            else mask

        for step in (1, -1):
            start = 0 if step > 0 \
                else len(array) - 1

            stop = len(array) if step > 0 \
                else -1

            corner = (-1, -1)

            for index in range(start, stop, step):
                row = array[index]
                if np.any(row):
                    value_index = np.nonzero(row)[0][0]
                    corner = (value_index, index) if is_transposed \
                        else (index, value_index)

                    break  # первый угол найден

            corners.append(corner)

    return corners


def FindNotebookImage(image: np.ndarray) -> np.ndarray:
    """
    Преобразует изображение, чтобы получить вид сверху, 
    как если бы это была отсканированная тетрадь.

    Функция использует перспективное преобразование для выравнивания изображения,
    предполагая, что в изображении есть красные маркеры по углам, которые указывают
    область тетради.  
    Она ищет эти маркеры, определяет их координаты и использует их для преобразования перспективы.

    Args:
        image (np.ndarray): исходной изображение (матрица цветов).

    Returns:
        np.ndarray: выходное изображение.
    """

    red_high, red_low = (6.0, 255, 255), (0.0, 0, 0)
    height, width, _ = image.shape

    src_pos = np.float32([[elem[1], elem[0]]
                          for elem in FindCorners(
                              cv2.inRange(cv2.cvtColor(image,   # type: ignore
                                                       cv2.COLOR_RGB2HSV),
                                          red_low,              # type: ignore
                                          red_high))])          # type: ignore

    dst_pos = np.float32(
        [[0, 0],  # type: ignore
         [width, height],
         [0, height],
         [width, 0]]
    )

    return cv2.warpPerspective(image.copy(),
                               cv2.getPerspectiveTransform(
        src_pos, dst_pos),  # type: ignore
        (width, height)
    )
