from typing import Any, TypedDict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.axes import Axes
import matplotlib.pyplot as polt  # дань "уважения" ""легенде""
from matplotlib.ticker import AutoMinorLocator


class RGBHists(TypedDict):
  r: np.ndarray
  g: np.ndarray
  b: np.ndarray


def SetGrid(ax: Axes,
            n_locator: int = 10,
            minor_line_width: float = 0.2,
            major_line_width: float = 0.4):
  """
  Настраивает сетку на графике, добавляя минорные и мажорные деления.

  Args:
      ax (Axes): ось, на которой требуется настроить сетку.
      n_locator (int, optional): кол-во минорных делений между мажорными. Defaults to 10.
      minor_line_width (float, optional): толщина линий минорной сетки. Defaults to 0.2.
      major_line_width (float, optional): толщина линий мажорной сетки. Defaults to 0.4.
  """

  ax.xaxis.set_minor_locator(AutoMinorLocator(n_locator))
  ax.yaxis.set_minor_locator(AutoMinorLocator(n_locator))

  ax.grid(which='minor', linestyle='--', linewidth=minor_line_width)
  ax.grid(which='major', linewidth=major_line_width)


def IsRGBHists(to_plot: dict) -> bool:
  """
  Проверяет, является ли входной словарь гистограммами RGB-изображения.

  Args:
      to_plot (dict): словарь, который необходимо проверить.

  Returns:
      bool: True, если словарь соответствует структуре RGBHists, иначе False.
  """

  return (type(to_plot) == RGBHists) or (to_plot.keys() == {"r", "g", "b"})


def VerbosePlot(to_plot: Any,
                title: str = "",
                x_label: str = "",
                y_label: str = "",
                figure_size: tuple[float, float] = (10, 6),
                is_image: bool = False):
  """
  Отображает график данных, гибко обрабатывая различные типы входных данных.

  Args:
      to_plot (Any): данные для отображения.
      title (str, optional): заголовок графика. Defaults to "".
      x_label (str, optional): метка оси X. Defaults to "".
      y_label (str, optional): метка оси Y. Defaults to "".
      figure_size (tuple[float, float], optional): размер фигуры (ширина, высота). 
                                                   Defaults to (10, 6).
  """

  if to_plot is None or (isinstance(to_plot, np.ndarray) and to_plot.size == 0):
    return

  _, axs = polt.subplots(1, 1, figsize=figure_size)

  axs.set_title(title)
  axs.set_xlabel(x_label)
  axs.set_ylabel(y_label)

  if isinstance(to_plot, dict):
    if IsRGBHists(to_plot):
      for color in ("r", "g", "b"):
        axs.plot(to_plot[color],
                 color=color,
                 label=color.upper())

    else:
      for key, value in to_plot.items():
        axs.plot(value, label=key)

    axs.legend()

  else:
    if is_image:
      axs.imshow(to_plot)
      axs.axis("off")

    else:
      axs.plot(to_plot)

  SetGrid(axs)


def SIFTDescriptors(image_path: str) -> np.ndarray | None:
  """
  Вычисляет SIFT дескрипторы для изображения.

  Args:
      image_path (str): путь к изображению.

  Returns:
      np.ndarray | None: список SIFT дескрипторов, 
                         если изображение успешно обработано, иначе None.
  """

  image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

  if image is None:
    return None

  image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  mask = cv2.inRange(image_hsv, (35, 50, 50), (85, 255, 255))  # type: ignore
  gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) * mask

  sift = cv2.SIFT_create()  # type: ignore
  _, descs = sift.detectAndCompute(gray_image, None)

  return descs


def TrainKMeans(
    image_paths: list[str],
    clusters_amount: int,
    random_state: int = 42
) -> KMeans:
  """
  Обучает модель KMeans на SIFT дескрипторах из набора изображений.

  Args:
      image_paths (list[str]): список путей к изображениям для обучения.
      clusters_amount (int): количество кластеров KMeans.
      random_state (int, optional): значение для инициализации генератора случайных чисел
                                    (чтобы гарантировать воспроизводимость результатов). 
                                    Defaults to 42.

  Raises:
      ValueError: если не найдено ни одного SIFT дескриптора в обучающих изображениях.

  Returns:
      KMeans: обученная модель
  """

  all_descriptors = []

  for path in image_paths:
    descriptors = SIFTDescriptors(path)
    if descriptors is not None:
      all_descriptors.append(descriptors)

  if not all_descriptors:
    raise ValueError("No SIFT descriptors found in training images.")

  all_descriptors = np.vstack(all_descriptors)

  k_means = KMeans(n_clusters=clusters_amount, random_state=random_state)
  k_means.fit(all_descriptors)

  return k_means


def PredictAndArrayTransform(
    image_path: str,
    k_means: KMeans,
    clusters_amount: int,
    show_hists: bool = False
) -> np.ndarray:
  """
  Вычисляет гистограмму визуальных слов для изображения, 
  используя обученную модель KMeans.

  Args:
      image_path (str): путь к изображению.
      k_means (KMeans): обученная модель KMeans.
      clusters_amount (int): количество кластеров (визуальных слов).
      show_hists (bool): отображать ли predict гистограммы. Defaults to False.

  Returns:
      np.ndarray: нормализованная гистограмма визуальных слов в виде массива NumPy.
  """

  descriptors = SIFTDescriptors(image_path)
  if descriptors is None:
    return np.zeros(clusters_amount)

  visual_words = k_means.predict(descriptors)
  hist, bin_edges = np.histogram(visual_words,
                                 bins=clusters_amount,
                                 range=(0, clusters_amount))
  hist = hist.astype(np.float32)

  if hist.max() != 0:
    hist /= hist.max()

  if show_hists:
    _, (ax, ax_2) = polt.subplots(1, 2, figsize=(12, 6))
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title("Source image")
    ax.axis("off")

    ax_2.bar(bin_edges[:-1],
             hist,
             width=np.diff(bin_edges),
             color='skyblue',
             edgecolor='black')

    ax_2.grid(axis='y',
              alpha=0.75)
    ax_2.set_title("Hist Prediction")
    ax_2.set_ylim(0, 1)

    polt.tight_layout()
    polt.show()

  return hist


def Transform(kernel: np.ndarray,
              points: np.ndarray) -> np.ndarray:
  """
  Применяет аффинное преобразование к набору точек, 
  используя заданное ядро (матрицу).

  Args:
      kernel (np.ndarray): матрица преобразования.
      points (np.ndarray): массив координат точек (Nx3, где N - количество точек, а столбцы - x, y, 1).

  Returns:
      np.ndarray: массив преобразованных координат точек (Nx3).
  """

  transformed_points = (kernel @ points.T).T
  return np.divide(transformed_points.T, transformed_points[:, 2]).T


def FindRANSACOffset(points: np.ndarray,
                     transformed_points: np.ndarray) -> np.ndarray | None:
  """
  Оценивает смещение (offset) между двумя наборами точек 
  с использованием упрощенного подхода, похожего на RANSAC.  

  (он предполагает, что смещение является константой для всех точек)

  Args:
      points (np.ndarray): исходный набор точек (Nx3).
      transformed_points (np.ndarray): набор преобразованных точек (Nx3).

  Returns:
      np.ndarray: оптимальное смещение (offset) в формате NumPy array [dx, dy].
      (возвращает None, если не найдено подходящего смещения)
  """

  best_offset = None
  best_loss = np.inf

  for point, transformed_point in zip(points, transformed_points):
    offset = (transformed_point - point)[:2]

    transformation = np.eye(3)
    transformation[:2, 2] = offset

    tmp = Transform(transformation, points)
    tmp_loss = np.linalg.norm(tmp - transformed_points)
    if tmp_loss < best_loss:
      best_loss = tmp_loss
      best_offset = offset

  return best_offset


def GetShift(first: np.ndarray,
             second: np.ndarray
             #  , puzzle_dir
             ) -> np.ndarray:
  """
  Вычисляет смещение между двумя изображениями, 
  используя алгоритмы SIFT и RANSAC.

  Args:
      first (np.ndarray): первое изображение.
      second (np.ndarray): второе изображение.

  Returns:
      np.ndarray: смещение между двумя изображениями в формате NumPy array [dy, dx]. 
      (Смещение указывает, 
      насколько нужно сдвинуть второе изображение, чтобы оно совпадало с первым)
  """

  hyp_params_1: dict
  hyp_params_2: dict

  # match puzzle_dir:
  #   case 'puzzle/su_fighter_shuffle' | 'puzzle/home_shuffle':
  #     pass
  #   case 'puzzle/su_fighter_shuffle':
  #     pass

  hyp_params_2 = dict(nfeatures=500,
                      nOctaveLayers=11,
                      contrastThreshold=0.03,
                      edgeThreshold=10,
                      sigma=1.7)

  hyp_params_1 = dict(nfeatures=first.size // second.size * 500,
                      nOctaveLayers=11,
                      contrastThreshold=0.03,
                      edgeThreshold=10,
                      sigma=1.7)

  sift_2 = cv2.SIFT_create(**hyp_params_2)
  sift_1 = cv2.SIFT_create(**hyp_params_1)

  FLANN_INDEX_KDTREE = 2
  index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                      trees=15)
  search_params = dict(checks=150)

  flann = cv2.FlannBasedMatcher(index_params,
                                search_params)

  k_points_1, descriptors_1 = sift_1.detectAndCompute(first, None)
  k_points_2, descriptors_2 = sift_2.detectAndCompute(second, None)

  matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
  ratio_thresh = 0.4
  good_matches = []

  for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
      good_matches.append(m)

  points = np.array(
    [[k_points_1[m.queryIdx].pt[1],
      k_points_1[m.queryIdx].pt[0], 1] for m in good_matches])

  transformed_points = np.array(
    [[k_points_2[m.trainIdx].pt[1],
      k_points_2[m.trainIdx].pt[0], 1] for m in good_matches])

  shift = FindRANSACOffset(transformed_points, points)

  if shift is not None:
    shift[0] = round(shift[0])
    shift[1] = round(shift[1])

  return np.array(shift, dtype=np.int32)


def StitchImagesRD(left: np.ndarray,
                   right: np.ndarray,
                   shift: np.ndarray) -> np.ndarray:
  """
  Сшивает два изображения,
  сдвигая правое изображение относительно левого, 
  используя заданное смещение.

  Args:
      left (np.ndarray): левое изображение.
      right (np.ndarray): правое изображение.
      shift (np.ndarray): смещение между двумя изображениями в формате NumPy array [dy, dx].

  Returns:
      np.ndarray: сшитое изображение.
  """

  new_shape = [max(left.shape[0], right.shape[0] + shift[0]),
               max(left.shape[1], right.shape[1] + shift[1]), 3]

  new_image = np.zeros(shape=new_shape)

  new_image[0:left.shape[0], 0:left.shape[1], :] = left
  new_image[shift[0]:right.shape[0] + shift[0],
            shift[1]:right.shape[1] + shift[1], :] = right

  new_image = new_image.astype(np.uint8)

  return new_image


def StitchImages(first: np.ndarray,
                 second: np.ndarray,
                 shift: np.ndarray) -> np.ndarray:
  """
  Сшивает два изображения, 
  сначала корректируя смещение, а затем сшивая изображения.

  Args:
      first (np.ndarray): первое изображение.
      second (np.ndarray): второе изображение.
      shift (np.ndarray): смещение между двумя изображениями в формате NumPy array [dy, dx].

  Returns:
      np.ndarray: сшитое изображение.
  """

  if shift[0] >= 0 and shift[1] < 0:
    shift[1] *= -1

    first = np.pad(first, ((0, 0), (shift[1], 0), (0, 0)))

    shift[1] = 0

  elif shift[0] < 0 and shift[1] >= 0:
    shift[0] *= -1

    first = np.pad(first, ((shift[0], 0), (0, 0), (0, 0)))

    shift[0] = 0

  elif shift[0] < 0 and shift[1] < 0:
    shift[0] *= -1
    shift[1] *= -1

    first = np.pad(first, ((shift[0], 0), (shift[1], 0), (0, 0)))

    shift[0] = 0
    shift[1] = 0

  return StitchImagesRD(first, second, shift)
