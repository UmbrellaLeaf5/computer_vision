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


def Puzzle(images_paths: list):
  """
  Собирает изображения в "пазл", используя SIFT для обнаружения общих признаков 
  и аффинные преобразования для выравнивания и слияния изображений.

  Args:
      images_paths: список путей к изображениям, которые нужно объединить.
  """

  n_features = 1000
  n_octave_layers = 3
  contrast_threshold = 0.04
  edge_threshold = 8
  sigma = 1.3

  detector = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers,
                             contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold,
                             sigma=sigma)

  # для улучшения контрастности
  clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
  flann = cv2.BFMatcher(cv2.NORM_L2, False)  # для сопоставления дескрипторов
  image_base = cv2.imread(images_paths[0])

  num_images = len(images_paths)
  grid_size = int(np.sqrt(num_images))
  H, W = (image_base.shape[0] * 2 * grid_size,
          image_base.shape[1] * 2 * grid_size)

  final = np.zeros((H, W, 3))
  row_start = H // 2 - image_base.shape[0] // 2
  row_end = H // 2 + image_base.shape[0] // 2
  col_start = W // 2 - image_base.shape[1] // 2
  col_end = W // 2 + image_base.shape[1] // 2
  final[row_start:row_end, col_start:col_end] = image_base / 255
  del image_base

  # хранит аффинные преобразования относительно базового изображения
  bases = [[] for _ in range(num_images)]
  candidates = list(range(1, num_images))  # идентификаторы необработанных изображений
  center = [0]  # идентификаторы изображений, добавленных в пазл
  checked = []  # идентификаторы изображений, которые были проверены на соответствие
  ratio_thresh = 0.47  # порог отношения расстояний для отбора хороших соответствий

  while center:
    i = center.pop()
    loc_image_i = cv2.imread(images_paths[i])

    yrcb_i = cv2.cvtColor(loc_image_i, cv2.COLOR_RGB2YCrCb)
    bright_i = clahe.apply(yrcb_i[:, :, 0])
    yrcb_i[:, :, 0] = bright_i
    loc_image_i = cv2.cvtColor(yrcb_i, cv2.COLOR_YCrCb2RGB)

    # обнаружение ключевых точек и вычисление дескрипторов SIFT
    keypoints_i, descriptors_i = detector.detectAndCompute(loc_image_i, None)

    # если дескрипторы не найдены, переходим к следующему изображению
    if descriptors_i is None:
      continue

    # поиск соответствий с другими изображениями
    while candidates:
      j = candidates.pop()
      checked.append(i)

      loc_image_j = cv2.imread(images_paths[j])

      yrcb_j = cv2.cvtColor(loc_image_j, cv2.COLOR_RGB2YCrCb)
      bright_j = clahe.apply(yrcb_j[:, :, 0])
      yrcb_j[:, :, 0] = bright_j
      loc_image_j = cv2.cvtColor(yrcb_j, cv2.COLOR_YCrCb2RGB)

      # обнаружение ключевых точек и вычисление дескрипторов SIFT
      keypoints_j, descriptors_j = detector.detectAndCompute(loc_image_j, None)

      # если дескрипторы не найдены, переходим к следующему изображению
      if descriptors_j is None:
        continue

      # поиск соответствий между дескрипторами
      raw_matches = flann.knnMatch(np.asarray(descriptors_i, np.float32),
                                   np.asarray(descriptors_j, np.float32), k=2)

      # отбор хороших соответствий на основе отношения расстояний
      good_matches = [m for m, n in raw_matches if len(
        raw_matches[0]) >= 2 and m.distance < ratio_thresh * n.distance]

      # найдено достаточно хороших соответствий ---
      if len(good_matches) >= 3:
        # определение координат соответствующих точек
        points1 = np.float32([keypoints_i[match.queryIdx].pt for match in good_matches])
        points2 = np.float32([keypoints_j[match.trainIdx].pt for match in good_matches])

        # создание маски для области, где будет добавлено новое изображение
        mask = np.uint8(cv2.cvtColor(final.astype(np.float32) * 255, cv2.COLOR_BGR2GRAY) == 0)
        # эрозия для уменьшения шума
        mask = cv2.erode(mask, np.ones((3, 3)), iterations=2)
        # расширение для восстановления формы
        mask = cv2.dilate(mask, np.ones((3, 3)), iterations=2)

        out_affine = cv2.estimateAffine2D(points2, points1)

        if out_affine[0] is not None:
          center.append(j)

          transformed_image = np.zeros((H, W, 3))
          row_start = H // 2 - loc_image_j.shape[0] // 2
          row_end = H // 2 + loc_image_j.shape[0] // 2
          col_start = W // 2 - loc_image_j.shape[1] // 2
          col_end = W // 2 + loc_image_j.shape[1] // 2
          transformed_image[row_start:row_end, col_start:col_end] = loc_image_j / 255

          for base in bases[i]:
            bases[j].append(base)
            transformed_image = cv2.warpAffine(transformed_image, base, (W, H))

          bases[j].append(out_affine[0])

          # применяем преобразование к изображению
          transformed_image = cv2.warpAffine(transformed_image, out_affine[0], (W, H))

          # накладываем преобразованное изображение на результирующее
          final += np.repeat(mask[..., np.newaxis], 3, axis=-1) * transformed_image

    candidates = [k for k in range(1, num_images) if k not in center and k not in checked]

  VerbosePlot(final, is_image=True)
