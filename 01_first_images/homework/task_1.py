from typing import Literal
import cv2
import numpy as np
import enum


Point = tuple[int, int]
ColorTuple = tuple[int, int, int]
ColorList = list[int]


class WallType(enum.Enum):
    NO_WALL = 0
    GREEN = 1
    BLACK = 2


class Direction(enum.Enum):
    DOWN = 0
    UP = 1
    LEFT = 2
    RIGHT = 3


WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
GREEN = [0, 255, 0]


def FindMazeEntryAndExit(image: np.ndarray) -> tuple[Point, Point]:
    """
    Находит вход и выход из лабиринта на изображении.
    (вход и выход расположены только сверху и снизу лабиринта)

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        tuple[Point, Point]: координаты входа и выхода лабиринта в формате (x, y).
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


def FloodFillFromPoint(image: np.ndarray, point: Point,
                       color: ColorTuple) -> np.ndarray:
    """
    Выполняет заливку области (flood fill) на изображении, начиная с указанной точки.

    Args:
        image (np.ndarray): изображение (матрица цветов).
        point (Point): координаты начальной точки заливки в формате (x, y).
        color (ColorTuple): цвет для заливки в формате кортежа (B, G, R).

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


def GetMazeCellSize(image: np.ndarray) -> int:
    # кол-во белых пикселей на входе + 2
    return len(np.where(np.all(image[0] == WHITE, axis=1))[0]) + 2


def GetCenterOfMazeCell(point: Point,
                        image: np.ndarray,
                        maze_cell_size: int = 0  # 0 - неизвестно
                        ) -> Point:
    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    h, w, _ = image.shape
    x, y = point

    cell_x = (x // maze_cell_size) * maze_cell_size + maze_cell_size // 2
    cell_y = (y // maze_cell_size) * maze_cell_size + maze_cell_size // 2

    return (min(cell_x, w - 2), min(h - 2, cell_y))  # во избежание выхода за границу


def GetWallStatus(point: Point,
                  image: np.ndarray,
                  maze_cell_size: int = 0  # 0 - неизвестна
                  ) -> dict[Direction, tuple[Point | None, WallType]]:
    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    x_c, y_c = GetCenterOfMazeCell(point, image, maze_cell_size)
    offset = maze_cell_size - 2
    h, w, _ = image.shape

    def GetNeighbor(dx, dy) -> Point | None:
        if x_c + dx > 0 and y_c + dy > 0:
            return GetCenterOfMazeCell((x_c + dx, y_c + dy), image, maze_cell_size)
        else:
            return None

    wall_status = {
        Direction.DOWN:  (GetNeighbor(0, offset),  WallType.NO_WALL),
        Direction.UP:    (GetNeighbor(0, -offset), WallType.NO_WALL),
        Direction.LEFT:  (GetNeighbor(-offset, 0), WallType.NO_WALL),
        Direction.RIGHT: (GetNeighbor(offset, 0),  WallType.NO_WALL)
    }

    def GetWallTypeFromColor(color: ColorList) -> WallType:
        if np.all(np.equal(color, BLACK)):
            return WallType.BLACK

        elif np.all(np.equal(color, GREEN)):
            return WallType.GREEN

        return WallType.NO_WALL

    for direction, (dx, dy) in {
        # список попиксельных смещений

        Direction.DOWN:  (0, 1),
        Direction.UP:    (0, -1),
        Direction.LEFT:  (-1, 0),
        Direction.RIGHT: (1, 0)
    }.items():
        for pix in range(1, maze_cell_size // 2 + 1):
            # масштабирование смещений
            x = x_c + dx * pix
            y = y_c + dy * pix

            if x >= 0 and x < w and y >= 0 and y < h:
                if np.any(np.not_equal(image[y, x], WHITE)):
                    wall_status[direction] = ((x, y),
                                              GetWallTypeFromColor(image[y, x]))
                    break  # нашли стену, переходим к следующему направлению

    return wall_status


def FindMazePath(image: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Находит путь через лабиринт.

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        tuple: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат.
    """

    answer: list[Point] = []

    cell_size = GetMazeCellSize(image)

    # заливка левой части лабиринта в зеленый
    image = FloodFillFromPoint(image, (0, 0), tuple(GREEN))  # type: ignore

    # высота лабиринта
    h, _, _ = image.shape

    # точки входа и выхода соотв.
    entry_point, exit_point = FindMazeEntryAndExit(image)
    answer.append(entry_point)

    # очевидно, что первой точкой после входа будет поход вниз:
    first_point = GetCenterOfMazeCell((entry_point[0], 4), image, cell_size)

    # предпоследней точкой будет верх от выхода:
    last_point = GetCenterOfMazeCell((exit_point[0], h - 4), image, cell_size)

    def GetNeighbors(point: Point) -> dict[Direction, Point | None]:
        res: dict[Direction, Point | None] = {
            Direction.DOWN: None,
            Direction.UP: None,
            Direction.LEFT: None,
            Direction.RIGHT: None
        }

        status = GetWallStatus(point, image, cell_size)

        for direction, point_wall in status.items():
            p, w = point_wall
            if w == WallType.NO_WALL:
                res[direction] = p

        return res

    def IsAcceptedNeighbor(point: Point,
                           direction: Direction) -> bool:

        def CalcWallDirection(wall_type: WallType) -> Direction:
            # autopep8: off
            match wall_type:
                case WallType.GREEN:
                    match direction:
                        case Direction.DOWN: return Direction.LEFT
                        case Direction.UP: return Direction.RIGHT
                        case Direction.LEFT: return Direction.UP
                        case Direction.RIGHT: return Direction.DOWN

                case WallType.BLACK:
                    match direction:
                        case Direction.DOWN: return Direction.RIGHT
                        case Direction.UP: return Direction.LEFT
                        case Direction.LEFT: return Direction.DOWN
                        case Direction.RIGHT: return Direction.UP
            # autopep8: on
            return Direction.UP  # невозможный случай

        wall_status = GetWallStatus(point, image, cell_size)

        green_wall_dir = CalcWallDirection(WallType.GREEN)
        black_wall_dir = CalcWallDirection(WallType.BLACK)

        def CheckWallType(wall_dir: Direction,
                          target_wall_type: WallType) -> bool:
            current_status = wall_status

            while True:
                if current_status[wall_dir][1] == target_wall_type:
                    return True
                elif current_status[wall_dir][1] == WallType.NO_WALL:
                    next_neighbor = current_status[wall_dir][0]
                    current_status = GetWallStatus(
                        next_neighbor, image, cell_size)  # type: ignore
                else:
                    return False  # стена другого цвета

        return CheckWallType(green_wall_dir, WallType.GREEN) and\
            CheckWallType(black_wall_dir, WallType.BLACK)

    def GetUniqueNeighbor(neighbors: dict[Direction, Point | None]) \
            -> Point | Literal[False]:
        unique_neighbor: Point | Literal[False] = False

        count = 0
        for neighbor in neighbors.values():
            # также сразу избегаем вариантов, которые уже в ответе
            if neighbor is not None and neighbor not in answer:
                count += 1

                if count > 1:
                    return False  # нашли второго соседа, сразу выходим

                unique_neighbor = neighbor  # type: ignore

        if count == 0:  # тупиковая ситуация
            return False
        else:
            return unique_neighbor

    def GetNextMazePathPoint(current_point: Point) -> Point:
        neighbors = GetNeighbors(current_point)

        # перебор прямых соседей
        for direction, neighbor in neighbors.items():
            if neighbor and neighbor not in answer and IsAcceptedNeighbor(neighbor, direction):
                return neighbor

        # перебор соседей соседей (2-step lookahead)
        for direction, neighbor in neighbors.items():
            if neighbor and neighbor not in answer:
                neighbors_2 = GetNeighbors(neighbor)
                for direction_2, neighbor_2 in neighbors_2.items():
                    if neighbor_2 and neighbor_2 not in answer and IsAcceptedNeighbor(neighbor_2, direction_2):
                        return neighbor

        return (-1, -1)  # невозможный исход (в нормальных лабиринтах)

    curr_point = first_point
    while (curr_point != last_point):
        answer.append(curr_point)

        neighbors = GetNeighbors(curr_point)
        unique_neighbor = GetUniqueNeighbor(neighbors)

        if unique_neighbor:
            curr_point = unique_neighbor
            continue  # развилки нет - идём дальше

        curr_point = GetNextMazePathPoint(curr_point)

    answer.append(last_point)
    answer.append(exit_point)

    return ([x for x, _ in answer], [y for _, y in answer])
