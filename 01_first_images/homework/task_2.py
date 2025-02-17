import cv2
import numpy as np


def FindRoadsEndpoints(road_image: np.ndarray) -> list[tuple[int, int]]:
    """
    Находит x-координаты первого и последнего пикселей во всех дорогах.

    Args:
        road_image (np.ndarray): изображение с выделенными дорогами.

    Returns:
        list[tuple[int, int]]: список с x-координатами начала и конца дорог.
    """

    roads_endpoints: list[tuple[int, int]] = []
    road_size: int = 0

    for i in range(road_image.shape[1]):
        if road_image[0][i]:
            road_size += 1

        else:
            if road_size != 0:
                roads_endpoints.append((i - road_size, i))
                road_size = 0

    # если дорога заканчивается в конце изображения
    if road_size != 0:
        roads_endpoints.append((road_image.shape[1] - road_size, road_image.shape[1]))

    return roads_endpoints


def GetBlockedRoadsIndexes(blocked_roads_image: np.ndarray,
                           roads_endpoints: list[tuple[int, int]]
                           ) -> list[int]:
    """
    Определяет индексы дорог, заблокированных препятствиями.

    Args:
        blocked_roads_image (np.ndarray): изображение с выделенными препятствиями.
        roads_positions (list[tuple[int, int]]): координаты начала и конца дорог.

    Returns:
        list[int]: список с номерами дорог, которые заблокированы препятствиями.
    """

    road_pixels = [(start + end) // 2 for start, end in roads_endpoints]  # Вычисляем пиксели дорог

    # bool массив, указывающий, заблокирован ли пиксель дороги
    blocked_status = np.any(blocked_roads_image[:, road_pixels], axis=0)

    blocked_roads_indexes = np.where(blocked_status)[0].tolist()
    return blocked_roads_indexes  # type: ignore


def FindRoadIndex(image: np.ndarray) -> int | None:
    """
    Находит индекс дороги, на которой нет препятствия в конце пути.

    Args:
        image (np.ndarray): исходное изображение (матрица цветов).

    Returns:
        int | None: индекс дороги, на котором нет препятствия на дороге.
                    None, если на всех есть препятствия.
    """

    grey_high, grey_low = (180.0, 18, 230), (0.0, 0,  40)
    red_high,  red_low = (9.0,   255, 255), (0.0, 50, 70)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    roads_image = cv2.inRange(image_hsv, grey_low, grey_high)        # type: ignore
    blocked_roads_image = cv2.inRange(image_hsv, red_low, red_high)  # type: ignore

    roads_positions = FindRoadsEndpoints(roads_image)
    obstacle_positions = GetBlockedRoadsIndexes(blocked_roads_image, roads_positions)

    for road_index in range(len(roads_positions)):
        if road_index not in obstacle_positions:
            return road_index

    return None
