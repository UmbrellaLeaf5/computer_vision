from glob import glob

import cv2
import numpy as np
from sklearn.cluster import KMeans


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
    clusters_amount: int
) -> np.ndarray:
  """
  Вычисляет гистограмму визуальных слов для изображения, 
  используя обученную модель KMeans.

  Args:
      image_path (str): путь к изображению.
      k_means (KMeans): обученная модель KMeans.
      clusters_amount (int): количество кластеров (визуальных слов).

  Returns:
      np.ndarray: нормализованная гистограмма визуальных слов в виде массива NumPy.
  """

  descriptors = SIFTDescriptors(image_path)
  if descriptors is None:
    return np.zeros(clusters_amount)

  visual_words = k_means.predict(descriptors)
  hist, _ = np.histogram(visual_words, bins=clusters_amount, range=(0, clusters_amount))
  hist = hist.astype(np.float32)

  if hist.sum() != 0:
    hist /= hist.sum()

  return hist
