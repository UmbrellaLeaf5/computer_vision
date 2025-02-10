from typing import Literal
import cv2
import numpy as np
import enum


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
                        image: np.ndarray,
                        maze_cell_size: int = 0  # 0 - неизвестно
                        ) -> tuple[int, int]:
    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    h, w, _ = image.shape

    x, y = point

    cell_x = (x // maze_cell_size) * maze_cell_size + maze_cell_size // 2
    cell_x = min(cell_x, w - 2)

    cell_y = (y // maze_cell_size) * maze_cell_size + maze_cell_size // 2
    cell_y = min(h - 2, cell_y)

    return (cell_x, cell_y)


def GetWallStatusFromPoint(point: tuple[int, int], image: np.ndarray,
                           maze_cell_size: int = 0  # 0 - неизвестна
                           ) -> dict[Direction, tuple[tuple[int, int] | None, WallType]]:

    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    x_c, y_c = GetCenterOfMazeCell(point, image, maze_cell_size)
    offset = maze_cell_size - 2

    def GetNeighbor(dx, dy):
        if x_c + dx > 0 and y_c + dy > 0:
            return GetCenterOfMazeCell((x_c + dx, y_c + dy), image, maze_cell_size)

        else:
            return None

    wall_status = {
        # (center(point), color)
        Direction.DOWN: (GetNeighbor(0, offset),
                         WallType.NO_WALL),

        Direction.UP: (GetNeighbor(0, -offset),
                       WallType.NO_WALL),

        Direction.LEFT: (GetNeighbor(-offset, 0),
                         WallType.NO_WALL),

        Direction.RIGHT: (GetNeighbor(offset, 0),
                          WallType.NO_WALL)


    }

    is_gotten = {
        Direction.DOWN: False,
        Direction.UP: False,
        Direction.LEFT: False,
        Direction.RIGHT: False
    }

    def GetWallTypeFromColor(color: list[int]):
        if np.all(np.equal(color, BLACK)):
            return WallType.BLACK

        elif np.all(np.equal(color, GREEN)):
            return WallType.GREEN

        return WallType.NO_WALL

    for pix in range(1, maze_cell_size // 2 + 1):
        if not is_gotten[Direction.DOWN]:
            if np.any(np.not_equal(image[y_c + pix, x_c], WHITE)):
                wall_status[Direction.DOWN] = (
                    (x_c, y_c + pix - 1), GetWallTypeFromColor(image[y_c + pix, x_c]))
                is_gotten[Direction.DOWN] = True

        if not is_gotten[Direction.UP] and (y_c - pix) > 0:
            if np.any(np.not_equal(image[y_c - pix, x_c], WHITE)):
                wall_status[Direction.UP] = (
                    (x_c, y_c - pix + 1), GetWallTypeFromColor(image[y_c - pix, x_c]))
                is_gotten[Direction.UP] = True

        if not is_gotten[Direction.LEFT] and (x_c - pix) > 0:
            if np.any(np.not_equal(image[y_c, x_c - pix], WHITE)):
                wall_status[Direction.LEFT] = (
                    (x_c - pix + 1, y_c), GetWallTypeFromColor(image[y_c, x_c - pix]))
                is_gotten[Direction.LEFT] = True

        if not is_gotten[Direction.RIGHT]:
            if np.any(np.not_equal(image[y_c, x_c + pix], WHITE)):
                wall_status[Direction.RIGHT] = (
                    (x_c + pix - 1, y_c), GetWallTypeFromColor(image[y_c, x_c + pix]))
                is_gotten[Direction.RIGHT] = True

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
    first_point = GetCenterOfMazeCell((entry_point[0], 4), image, cell_size)

    # предпоследней точкой будет верх от выхода:
    last_point = GetCenterOfMazeCell((exit_point[0], h - 4), image, cell_size)

    def GetNeighbors(point: tuple[int, int]) -> dict[Direction, tuple[int, int] | None]:
        res: dict[Direction, tuple[int, int] | None] = {
            Direction.DOWN: None,
            Direction.UP: None,
            Direction.LEFT: None,
            Direction.RIGHT: None
        }

        status = GetWallStatusFromPoint(point, image, cell_size)

        for direction, point_wall in status.items():
            p, w = point_wall
            if w == WallType.NO_WALL:
                res[direction] = p

        return res

    def IsAcceptedNeighbor(point: tuple[int, int],
                           direction: Direction) -> bool:

        def CalcWallDirection(direction: Direction,
                              wall_type: WallType) -> Direction:
            match wall_type:
                case WallType.GREEN:
                    match direction:
                        case Direction.DOWN:
                            return Direction.LEFT
                        case Direction.UP:
                            return Direction.RIGHT
                        case Direction.LEFT:
                            return Direction.UP
                        case Direction.RIGHT:
                            return Direction.DOWN

                case WallType.BLACK:
                    match direction:
                        case Direction.DOWN:
                            return Direction.RIGHT
                        case Direction.UP:
                            return Direction.LEFT
                        case Direction.LEFT:
                            return Direction.DOWN
                        case Direction.RIGHT:
                            return Direction.UP

            return Direction.UP

        walls_direction = {
            WallType.GREEN: CalcWallDirection(direction, WallType.GREEN),
            WallType.BLACK: CalcWallDirection(direction, WallType.BLACK)
        }

        wall_status = GetWallStatusFromPoint(point, image, cell_size)

        bool_dict = {
            WallType.GREEN: False,
            WallType.BLACK: False
        }

        for neigh_direction in wall_status.keys():
            for wall_type, wall_dir in walls_direction.items():
                if neigh_direction == wall_dir:
                    if wall_status[wall_dir][1] == wall_type:
                        bool_dict[wall_type] = True
                        break

                    elif wall_status[wall_dir][1] == WallType.NO_WALL:
                        copy_wall_status = wall_status.copy()

                        while (True):  # пока не найдем стенку соседа в том же направлении
                            next_neigh = copy_wall_status[wall_dir][0]

                            next_neigh_wall_status = \
                                GetWallStatusFromPoint(next_neigh, image, cell_size)  # type: ignore

                            if next_neigh_wall_status[wall_dir][1] == wall_type:
                                bool_dict[wall_type] = True
                                break

                            elif copy_wall_status[wall_dir][1] == WallType.NO_WALL:
                                copy_wall_status = next_neigh_wall_status

                            else:  # стена другого цвета
                                break

                    else:  # стена другого цвета
                        break

        return bool_dict[WallType.BLACK] and bool_dict[WallType.GREEN]

    def GetUniqueNeighbor(neigh_points: dict[Direction, tuple[int, int] | None]) \
            -> tuple[Direction, tuple[int, int]] | Literal[False]:
        unique_neigh: tuple[Direction, tuple[int, int]] | Literal[False] = False

        count = 0

        for item in neigh_points.items():
            _, neigh_point = item
            if neigh_point is not None and neigh_point not in answer:
                count += 1

                if count > 1:
                    return False  # нашли второе non-None, сразу выходим

                unique_neigh = item  # type: ignore

        if count == 0:
            return False
        else:
            return unique_neigh

    curr_point = first_point
    while (curr_point != last_point):
        answer.append(curr_point)

        wall_status = GetWallStatusFromPoint(curr_point, image, cell_size)
        neigh_points = GetNeighbors(curr_point)
        unique_neigh = GetUniqueNeighbor(neigh_points)

        prev_point = curr_point

        if unique_neigh:
            _, curr_point = unique_neigh

        if prev_point == curr_point:
            for direction, neigh_point in neigh_points.items():
                if neigh_point and neigh_point not in answer:
                    if IsAcceptedNeighbor(neigh_point, direction):
                        curr_point = neigh_point
                        break

        if prev_point == curr_point:
            for _, neigh_point in neigh_points.items():
                if neigh_point and neigh_point not in answer:
                    neigh_neigh_points = GetNeighbors(neigh_point)

                    for neigh_direction, neigh_neigh_point in neigh_neigh_points.items():
                        if neigh_neigh_point and neigh_neigh_point not in answer:
                            if IsAcceptedNeighbor(neigh_neigh_point, neigh_direction):
                                curr_point = neigh_point
                                break

    answer.append(last_point)
    answer.append(exit_point)

    return ([x for x, _ in answer], [y for _, y in answer])
