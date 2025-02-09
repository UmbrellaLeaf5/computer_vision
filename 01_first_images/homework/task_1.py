import cv2
import numpy as np
import enum


class WallType(enum.Enum):
    NO_WALL = 0
    GREEN = 1
    BLACK = 2


WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
GREEN = [0, 255, 0]


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
            np.all(image[row_index] == WHITE, axis=1))[0]

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
    return len(np.where(np.all(image[0] == WHITE, axis=1))[0]) + 2


def GetCenterOfMazeCell(point: tuple[int, int],
                        maze_cell_size: int) -> tuple[int, int]:
    x, y = point

    return ((x // maze_cell_size) * maze_cell_size + maze_cell_size // 2,
            (y // maze_cell_size) * maze_cell_size + maze_cell_size // 2)


def GetWallStatusFromPoint(point: tuple[int, int], image: np.ndarray,
                           maze_cell_size: int = 0  # 0 - неизвестна
                           ) -> dict[str, tuple[tuple[int, int] | None, WallType]]:

    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    x_c, y_c = GetCenterOfMazeCell(point, maze_cell_size)

    offset = maze_cell_size - 2

    def GetNeighbor(dx, dy):
        if x_c + dx > 0 and y_c + dy > 0:
            return GetCenterOfMazeCell((x_c + dx, y_c + dy), maze_cell_size)

        else:
            return None

    wall_status = {
        # (center(point), color)
        "down": (GetNeighbor(0, offset),
                 WallType.NO_WALL),

        "up": (GetNeighbor(0, -offset),
               WallType.NO_WALL),

        "right": (GetNeighbor(offset, 0),
                  WallType.NO_WALL),

        "left": (GetNeighbor(-offset, 0),
                 WallType.NO_WALL)
    }

    is_gotten = {
        "down": False,
        "up": False,
        "right": False,
        "left": False
    }

    def GetWallTypeFromColor(color: list[int]):
        if np.all(np.equal(color, BLACK)):
            return WallType.BLACK

        elif np.all(np.equal(color, GREEN)):
            return WallType.GREEN

        return WallType.NO_WALL

    for pix in range(1, offset):
        if np.any(np.not_equal(image[y_c + pix, x_c], WHITE)) and not is_gotten["down"]:
            wall_status["down"] = ((x_c, y_c + pix), GetWallTypeFromColor(image[y_c + pix, x_c]))
            is_gotten["down"] = True

        if np.any(np.not_equal(image[y_c - pix, x_c], WHITE)) and not is_gotten["up"] and (y_c - pix) > 0:
            wall_status["up"] = ((x_c, y_c - pix), GetWallTypeFromColor(image[y_c - pix, x_c]))
            is_gotten["up"] = True

        if np.any(np.not_equal(image[y_c, x_c + pix], WHITE)) and not is_gotten["right"]:
            wall_status["right"] = ((x_c + pix, y_c), GetWallTypeFromColor(image[y_c, x_c + pix]))
            is_gotten["right"] = True

        if np.any(np.not_equal(image[y_c, x_c - pix], WHITE)) and not is_gotten["left"] and (x_c - pix) > 0:
            wall_status["left"] = ((x_c - pix, y_c), GetWallTypeFromColor(image[y_c, x_c - pix]))
            is_gotten["left"] = True

    return wall_status


def FindWayFromMaze(image: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Находит путь через лабиринт.

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        tuple: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат.
    """

    answer: list[tuple[int, int]] = []

    cell_size = GetMazeCellSize(image)

    image = FloodFillFromPoint(image, (0, 0), tuple(GREEN))  # type: ignore
    # grid_image = AddGrid(image, step_size=cell_size, line_thickness=2, line_color=(100, 100, 100))

    h, _, _ = image.shape

    entry_point, exit_point = FindMazeEntryAndExit(image)

    answer.append(entry_point)

    # очевидно, что первой точкой после входа будет поход вниз:
    first_point = GetCenterOfMazeCell((entry_point[0], 4), cell_size)
    answer.append(first_point)

    # предпоследней точкой будет верх от выхода:
    last_point = GetCenterOfMazeCell((exit_point[0], h - 4), cell_size)

    def GetNeighbors(point: tuple[int, int]) -> dict[str, tuple[int, int] | None]:
        res: dict[str, tuple[int, int] | None] = {
            "down": None,
            "up": None,
            "right": None,
            "left": None
        }

        status = GetWallStatusFromPoint(point, image, cell_size)

        for direction, point_wall in status.items():
            p, w = point_wall
            if w == WallType.NO_WALL:
                res[direction] = p

        return res

    i = 0
    curr_point = first_point
    while (curr_point != last_point):
        print(curr_point)
        print(GetWallStatusFromPoint(curr_point, image, cell_size))
        print(GetNeighbors(curr_point))

        i += 1
        print("i: ", i)
        if i == 1:
            break

    answer.append(last_point)
    answer.append(exit_point)

    print(answer)

    return ([x for x, _ in answer], [y for _, y in answer])
