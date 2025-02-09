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

    h, _, _ = image.shape

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

    entry_x: int = (up_endpoints[0] + up_endpoints[1]) // 2
    exit_x: int = (down_endpoints[0] + down_endpoints[1]) // 2

    return ((entry_x, 0), (exit_x, h - 1))


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

    h, w, _ = image.shape

    if not (0 <= point[0] < w and 0 <= point[1] < h):
        raise ValueError("FloodFillFromPoint: point is outside image boundaries.")

    ff_image = image.copy()

    # маска (на 2 пикселя больше изображения)
    mask = np.zeros((h+2, w+2), np.uint8)

    _, ff_image, _, _ = cv2.floodFill(ff_image, mask, point, color)

    return ff_image


def AddGrid(image: np.ndarray, step_size: int,
            line_color: tuple[int, int, int] = (0, 0, 0), line_thickness: int = 1):

    for color_value in line_color:
        if not (0 <= color_value <= 255):
            raise ValueError(
                "AddGrid: RGB color values ​​must be in range(0, 256).")

    h, w, _ = image.shape

    grid_image = image.copy()

    for x in range(0, w, step_size):
        cv2.line(grid_image, (x, 0), (x, h), line_color, line_thickness)

    for y in range(0, h, step_size):
        cv2.line(grid_image, (0, y), (w, y), line_color, line_thickness)

    return grid_image


def GetMazeCellSize(image: np.ndarray) -> int:
    # кол-во белых пикселей на входе + 2
    return len(np.where(np.all(image[0] == [255, 255, 255], axis=1))[0]) + 2


def GetCenterOfMazeCell(point: tuple[int, int],
                        maze_cell_size: int) -> tuple[int, int]:
    x, y = point

    cell_x = (x // maze_cell_size) * maze_cell_size
    cell_y = (y // maze_cell_size) * maze_cell_size

    center_x = cell_x + maze_cell_size // 2
    center_y = cell_y + maze_cell_size // 2

    return (center_x, center_y)


def GetPositionStatus(image: np.ndarray,
                      point: tuple[int, int],
                      maze_cell_size: int = 0  # 0 - неизвестна
                      ) -> dict[str, tuple[int, list[int]]]:
    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    c_point = GetCenterOfMazeCell(point, maze_cell_size)
    x_c, y_c = c_point

    print(x_c, y_c)

    status = {
        "up": (0, [255, 255, 255]),      # (int, color)
        "down": (0, [255, 255, 255]),    # (int, color)
        "right": (0, [255, 255, 255]),   # (int, color)
        "left": (0, [255, 255, 255])     # (int, color)
    }

    is_gotten = {
        "up": False,
        "down": False,
        "right": False,
        "left": False
    }

    for pix in range(maze_cell_size // 2 + 2):
        if np.any(np.not_equal(image[y_c - pix, x_c], [255, 255, 255])) and not is_gotten["up"]:
            status["up"] = (pix, image[y_c - pix, x_c])
            is_gotten["up"] = True

        if np.any(np.not_equal(image[y_c + pix, x_c], [255, 255, 255])) and not is_gotten["down"] and (y_c - pix) > 0:
            status["down"] = (pix, image[y_c + pix, x_c])
            is_gotten["down"] = True

        if np.any(np.not_equal(image[y_c, x_c + pix], [255, 255, 255])) and not is_gotten["right"]:
            status["right"] = (pix, image[y_c, x_c + pix])
            is_gotten["right"] = True

        if np.any(np.not_equal(image[y_c, x_c - pix], [255, 255, 255])) and not is_gotten["left"] and (x_c - pix) > 0:
            status["left"] = (pix, image[y_c, x_c - pix])
            is_gotten["left"] = True

    return status


def FindWayFromMaze(image: np.ndarray) -> tuple[list[int], list[int]]:
    # -> list[tuple[list[int], list[int]] | np.ndarray]:
    """
    Находит путь через лабиринт.

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        tuple: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат.
    """

    x_list: list[int] = []
    y_list: list[int] = []

    def AddPointToAnswer(point: tuple[int, int]):
        x_list.append(point[0])
        y_list.append(point[1])

    cell_size = GetMazeCellSize(image)

    image = FloodFillFromPoint(image, (0, 0), (0, 255, 0))
    grid_image = AddGrid(image, step_size=cell_size, line_thickness=2, line_color=(100, 100, 100))

    h, w, _ = image.shape

    entry_point, exit_point = FindMazeEntryAndExit(image)

    AddPointToAnswer(entry_point)

    # очевидно, что первой точкой после входа будет поход вниз:
    first_point = GetCenterOfMazeCell((entry_point[0], 4), cell_size)
    AddPointToAnswer(first_point)

    # предпоследней точкой будет верх от выхода:
    last_point = GetCenterOfMazeCell((exit_point[0], h - 4), cell_size)

    curr_point = first_point
    while (curr_point != last_point):
        curr_status = GetPositionStatus(image, curr_point, cell_size)
        print(curr_point)
        print(curr_status)
        break

    AddPointToAnswer(last_point)
    AddPointToAnswer(exit_point)

    # image = FloodFillFromPoint(image, (0, 0), (0, 0, 0))

    # return [(x_list, y_list), image]

    print()

    print(GetPositionStatus(image, (5, 5)))
    print(GetPositionStatus(image, (5, h - 5)))

    print(GetPositionStatus(image, (w - 5, 5)))
    print(GetPositionStatus(image, (w - 5, h - 5)))

    return (x_list, y_list)
