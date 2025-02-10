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


WallStatus = dict[Direction, tuple[Point | None, WallType]]


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

    up_endpoints: tuple[int, int]
    down_endpoints: tuple[int, int]

    def FindWhiteEndpointsInRow(image: np.ndarray,
                                row_index: int
                                ) -> tuple[int, int]:
        """
        Находит x-координаты первого и последнего белых пикселей в заданной строке.

        Args:
            image (np.ndarray): изображение (матрица цветов) лабиринта.
            row_index (int): индекс строки для поиска.

        Returns:
            tuple[int, int]: кортеж x-координат первого и последнего белых пикселей.
            (возвращает пустой список, если белые пиксели не найдены)
        """

        white_pixels = np.where(
            np.all(image[row_index] == WHITE, axis=1))[0]

        if len(white_pixels) == 0:
            return (0, 0)
        return (white_pixels[0], white_pixels[-1])

    up_endpoints = FindWhiteEndpointsInRow(image, 0)
    down_endpoints = FindWhiteEndpointsInRow(image, h - 1)

    if not up_endpoints or not down_endpoints:
        raise ValueError("FindMazeEntryAndExit: enter or exit is not found.")

    entry_x: int = (up_endpoints[0] + up_endpoints[1]) // 2
    exit_x: int = (down_endpoints[0] + down_endpoints[1]) // 2

    return ((entry_x, 0), (exit_x, h - 1))


def FloodFillFromPoint(image: np.ndarray,
                       point: Point,
                       color: ColorTuple
                       ) -> np.ndarray:
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
    """
    Определяет размер ячейки лабиринта.
    (на основе количества белых пикселей в первой строке изображения)

    Args:
        image (np.ndarray): изображение (матрица цветов) лабиринта.

    Returns:
        int: размер ячейки лабиринта в пикселях.
    """

    # кол-во белых пикселей на входе + 2
    return len(np.where(np.all(image[0] == WHITE, axis=1))[0]) + 2


def GetCenterOfMazeCell(point: Point,
                        image: np.ndarray,
                        maze_cell_size: int = 0
                        ) -> Point:
    """
    Определяет координаты центра ячейки лабиринта.

    Args:
        point (Point): координаты точки внутри ячейки.
        image (np.ndarray): изображение (матрица цветов) лабиринта.
        maze_cell_size (int, optional): размер ячейки лабиринта.
                                        (если не указан, вычисляется автоматически)
                                        Defaults to 0.

    Returns:
        Point: координаты центра ячейки.
    """

    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    h, w, _ = image.shape
    x, y = point

    cell_x = (x // maze_cell_size) * maze_cell_size + maze_cell_size // 2
    cell_y = (y // maze_cell_size) * maze_cell_size + maze_cell_size // 2

    return (min(cell_x, w - 2), min(h - 2, cell_y))  # во избежание выхода за границу


def GetWallStatus(point: Point,
                  image: np.ndarray,
                  maze_cell_size: int = 0
                  ) -> WallStatus:
    """
    Определяет статус стен вокруг заданной точки в лабиринте.

    Для каждого направления (вверх, вниз, влево, вправо) функция определяет,
    есть ли стена, и если есть, то какого она типа (черная или зеленая).

    Args:
        point (Point): координаты точки, для которой определяется статус стен.
        image (np.ndarray): изображение (матрица цветов) лабиринта.
        maze_cell_size (int, optional): размер ячейки лабиринта.
                                        (если не указан, вычисляется автоматически)
                                        Defaults to 0.

    Returns:
        WallStatus: словарь, где 
        ключи - это направления (`Direction`), 
        значения - кортежи, содержащие координаты пикселя стены (`Point`) и тип стены (`WallType`).

        Если в данном направлении стены нет, то координата пикселя равна `None`, а тип стены `WallType.NO_WALL`.
    """

    if not maze_cell_size:
        maze_cell_size = GetMazeCellSize(image)

    x_c, y_c = GetCenterOfMazeCell(point, image, maze_cell_size)
    offset = maze_cell_size - 2
    h, w, _ = image.shape

    def GetNeighbor(dx: int,
                    dy: int
                    ) -> Point | None:
        """
        Возвращает координаты соседней ячейки.

        Вычисляет координаты соседней ячейки, смещенной на (dx, dy) от текущей,
        и возвращает центр этой ячейки.

        Args:
            dx (int): смещение по оси x.
            dy (int): смещение по оси y.

        Returns:
            OPoint | None: координаты центра соседней ячейки, или None, если соседняя ячейка находится за пределами изображения.
        """

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
        """
        Определяет тип стены на основе её цвета.

        Args:
            color (ColorList): цвет пикселя стены.

        Returns:
            WallType: тип стены (черная, зеленая, отсутствие стены).
        """

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
        """
        Определяет соседние ячейки для заданной точки.

        Для каждой ячейки возвращает её соседа в каждом из четырех направлений 
        (вверх, вниз, влево, вправо),
        если этот сосед существует и не является стеной.

        Args:
            point (Point): координаты ячейки, для которой нужно найти соседей.

        Returns:
            dict[Direction, Point | None]: словарь, где ключи - это направления (`Direction`),
            а значения - координаты соседней ячейки (`Point`) или None, если сосед отсутствует.
        """

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
                           direction: Direction
                           ) -> bool:
        """
        Определяет, является ли соседняя ячейка "приемлемой" для включения в путь.

        Ячейка считается приемлемой, если она находится рядом с зеленой и черной стенами
        в определенных направлениях относительно направления движения.
        (зеленая - всегда справа, черная - всегда слева)

        Args:
            point (Point): координаты соседней ячейки.
            direction (Direction): направление движения к этой ячейке.

        Returns:
            bool: True, если соседняя ячейка приемлема, иначе False.
        """

        def GetWallDirection(wall_type: WallType) -> Direction:
            """
            Вычисляет направление, в котором должна находиться стена,
            в зависимости от типа стены и направления движения.

            Args:
                wall_type (WallType): тип стены (зеленая или черная).

            Returns:
                Direction: направление, в котором должна находиться стена.
            """

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

        green_wall_dir = GetWallDirection(WallType.GREEN)
        black_wall_dir = GetWallDirection(WallType.BLACK)

        def IsAcceptedWallType(wall_dir: Direction,
                               wall_type: WallType
                               ) -> bool:
            """
            Проверяет, есть ли стена заданного типа в указанном направлении.

            Args:
                wall_dir (Direction): направление, в котором нужно проверить наличие стены.
                wall_type (WallType): тип стены, который нужно проверить (зеленая или черная).

            Returns:
                bool: True, если стена заданного типа есть в указанном направлении, иначе False.
            """

            current_status = wall_status

            while True:
                if current_status[wall_dir][1] == wall_type:
                    return True
                elif current_status[wall_dir][1] == WallType.NO_WALL:
                    next_neighbor = current_status[wall_dir][0]
                    current_status = GetWallStatus(
                        next_neighbor, image, cell_size)  # type: ignore
                else:
                    return False  # стена другого цвета

        return IsAcceptedWallType(green_wall_dir, WallType.GREEN) and\
            IsAcceptedWallType(black_wall_dir, WallType.BLACK)

    def FindUniqueNeighbor(neighbors: dict[Direction, Point | None]) \
            -> Point | Literal[False]:
        """
        Находит уникального соседа (если он есть) среди заданных соседей.

        Уникальным считается сосед, который является единственным возможным следующим шагом.
        (т.е., имеет только одного соседа, который еще не включен в путь)

        Returns:
            Point | Literal[False]: координаты уникального соседа, если он есть, иначе False.
        """

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
        """
        Находит следующую точку пути в лабиринте.

        Ищет следующую точку пути, перебирая соседей текущей точки
        и выбирая ту, которая является приемлемой (согласно функции IsAcceptedNeighbor)
        и еще не включена в путь.
        Если такого соседа нет, то ищет "соседей соседей".

        Args:
            current_point (Point): координаты текущей точки пути.

        Returns:
            Point: координаты следующей точки пути.
        """

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
        unique_neighbor = FindUniqueNeighbor(neighbors)

        if unique_neighbor:
            curr_point = unique_neighbor
            continue  # развилки нет - идём дальше

        curr_point = GetNextMazePathPoint(curr_point)

    answer.append(last_point)
    answer.append(exit_point)

    return ([x for x, _ in answer], [y for _, y in answer])
