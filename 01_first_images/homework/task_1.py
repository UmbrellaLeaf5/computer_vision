import cv2
import numpy as np


def FindMazeEntryAndExit(image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Находит вход и выход из лабиринта на изображении.
    (вход и выход расположены только сверху и снизу лабиринта)

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: координаты входа и выхода лабиринта в формате (x, y).
        ((x_entry, y_entry), (x_exit, y_exit))
    """

    h, _ = image.shape[:-1]

    up_endpoints: list[int] = []
    down_endpoints: list[int] = []

    def FindWhiteEndpointsInRow(image: np.ndarray, row_index: int) -> list[int]:
        """
        Находит x-координаты первой и последней белой пикселей в заданной строке.

        Args:
            image (np.ndarray): изображение (матрица цветов) лабиринта.
            row_index (int): индекс строки для поиска.

        Returns:
            list[int]: список x-координат первой и последней белых пикселей.
            (возвращает пустой список, если белые пиксели не найдены)
        """

        white_pixels = np.where(
            np.all(image[row_index] == [255, 255, 255], axis=1))[0]

        if len(white_pixels) == 0:
            return []
        return [white_pixels[0], white_pixels[-1]]

    up_endpoints = FindWhiteEndpointsInRow(image, 0)
    down_endpoints = FindWhiteEndpointsInRow(image, h - 1)

    if not up_endpoints or not down_endpoints:
        raise ValueError("FindMazeEntryAndExit: enter or exit is not found.")

    entry_i = int(0.5 * (up_endpoints[0] + up_endpoints[1]))
    exit_i = int(0.5 * (down_endpoints[0] + down_endpoints[1]))

    return ((entry_i, 0), (exit_i, h - 1))


def FloodFillFromPoint(image: np.ndarray, point: tuple[int, int],
                       color: tuple[int, int, int]) -> np.ndarray:
    """
    Выполняет заливку области (flood fill) на изображении, начиная с указанной точки.

    Args:
        image (np.ndarray): изображение (матрица цветов).
        point (tuple[int, int]): координаты начальной точки заливки в формате (x, y).
        color (tuple[int, int, int]): цвет для заливки в формате кортежа (B, G, R).

    Raises:
        ValueError: `FloodFillFromPoint: RGB color values ​​must be in range(0, 256).`
        ValueError: `FloodFillFromPoint: point is outside image boundaries.`

    Returns:
        np.ndarray: изображение с примененной заливкой области.
    """

    for color_value in color:
        if not (0 <= color_value <= 255):
            raise ValueError(
                "FloodFillFromPoint: RGB color values ​​must be in range(0, 256).")

    h, w = image.shape[:-1]

    if not (0 <= point[0] < w and 0 <= point[1] < h):
        raise ValueError("FloodFillFromPoint: point is outside image boundaries.")

    # Создаем маску (на 2 пикселя больше изображения)
    mask = np.zeros((h+2, w+2), np.uint8)

    _, image, _, _ = cv2.floodFill(image, mask, point, color)

    return image


def FindWayFromMaze(image: np.ndarray) -> tuple:
    """
    Находит путь через лабиринт.

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        tuple: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат.
    """

    coords = None
    # Ваш код тут

    image = FloodFillFromPoint(image, (0, 0), (0, 255, 0))

    h, w = image.shape[:-1]

    entry_point, exit_point = FindMazeEntryAndExit(image)

    for i in range(h):
        for j in range(w):
            pass

    image = FloodFillFromPoint(image, (0, 0), (0, 0, 0))

    return coords
