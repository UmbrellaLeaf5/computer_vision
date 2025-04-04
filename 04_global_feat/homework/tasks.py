import numpy as np
import os
from pathlib import Path
import sys
from typing import Any, Literal, TypedDict

import cv2
import numpy as np
import scipy as sc

import skimage
import sklearn

from matplotlib.axes import Axes
import matplotlib.pyplot as polt  # дань "уважения" ""легенде""
from matplotlib.ticker import AutoMinorLocator
from PIL import Image as PILImage


Hist = np.ndarray
Image = np.ndarray
Contour = np.ndarray


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


def PlotImages(images: list[str] | list[Image],
               title: str = "",
               hists_titles: list[str] | None = None,
               plot_only_hists: bool = False,
               figsize: tuple[int, int | None | Literal[0]] = (12, None),
               columns_amount: int = 3):
  """
  Отображает изображения или их RGB гистограммы в виде сетки графиков.

  Args:
      images (list[str]): список путей к изображениям для отображения или сами изображения.
      title (str, optional): общий заголовок для всей сетки графиков. Defaults to "".
      hists_titles (list[str] | None, optional): список заголовков для гистограмм 
                                                    (или изображений, если 
                                                    `plot_only_hists==False`).
                                                    Если None,
                                                    используются пути к изображениям. 
                                                    Defaults to None.
      plot_only_hists (bool, optional): если True, отображаются только гистограммы, 
                                        иначе - изображения. Defaults to False.
      image_scale (float, optional): коэффициент масштабирования изображений.
                                     По умолчанию 1.0 (оригинальный размер).
                                     Меньше 1.0 - уменьшение, больше 1.0 - увеличение.

  Raises:
      ValueError: Если длины списков `images` и `hists_titles` не совпадают.
  """

  amount = len(images)

  if amount == 0:
    return

  is_paths: bool = isinstance(images[0], str)
  image_paths: list[str] = images if is_paths else []  # type: ignore

  if is_paths:
    if hists_titles is None:
      hists_titles = image_paths  # используем пути, заголовки не предоставлены

    if amount != len(hists_titles):
      raise ValueError(
        "PlotImages: lengths of `image_paths` and `hists_titles` "
        f"do not match ({amount} and {len(hists_titles)}).")

  n_rows = (amount + columns_amount - 1) // columns_amount

  figsize = figsize if figsize[1] is not None and figsize[1] != 0 else (
    figsize[0], (figsize[0] // columns_amount) * n_rows)

  fig, axs = polt.subplots(n_rows, columns_amount, figsize=figsize)
  fig.suptitle(title, fontsize=24)

  for i, ax in enumerate(axs.flatten()):
    ax.axis("off")

    if i < amount:
      image = images[i]
      if hists_titles is not None:
        ax.set_title(hists_titles[i])

      if not plot_only_hists:
        try:
          if is_paths:
            image = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)

          ax.imshow(image)

        except Exception as exception:
          print("PlotImages: error loading or displaying for "
                f"`{image_paths[i] if is_paths else f"image_{i}"}`: "
                f"{exception}", file=sys.stderr)

      else:
        try:
          if is_paths:
            image = cv2.imread(image_paths[i])

          hist = GetRGBImageHists(image)  # type: ignore

          for color in ("r", "g", "b"):
            ax.plot(hist[color],
                    color=color,
                    label=color.upper())

            ax.legend()

          ax.set_xlabel("Pixel value")
          ax.set_ylabel("Frequency")
          ax.axis("on")

        except Exception as exception:
          print("PlotImages: error calculating or plotting histogram for "
                f"`{image_paths[i] if is_paths else f"image_{i}"}`: "
                f"{exception}", file=sys.stderr)

      SetGrid(ax)

  fig.tight_layout()


def GetImageContours(image: Image) -> list[Contour]:
  """
  Возвращает контуры изображения для конкретной задачи.

  Args:
      image (Image): исходное изображение (матрица цветов).

  Returns:
      list[Contour]: список контуров, найденных на изображении.
  """

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Contrast Limited Adaptive Histogram Equalization: для улучшения контраста
  clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(gray)

  # медианный фильтр для удаления шума
  blur = cv2.medianBlur(clahe, 9)

  # расширение (для соединения разрывов в контурах)
  dilate = cv2.dilate(blur, np.ones((5, 5), np.uint8), iterations=3)

  # двусторонний фильтр для сглаживания
  blur = cv2.bilateralFilter(dilate, d=9, sigmaColor=50, sigmaSpace=40)

  # пороговая обработка для бинаризации изображения
  th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

  # расширение (для улучшения контуров перед поиском)
  erode = cv2.dilate(th2, np.ones((3, 3), np.uint8), iterations=3)

  contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return list(contours)


def RightImagesBoolList(target_image: Image,
                        source_images: list[Image],
                        delta=0.15,
                        need_to_draw_contour: bool = False) -> list[bool]:
  """
  Сравнивает контур наибольшей площади на целевом изображении с контурами наибольшей площади
  на каждом из исходных изображений.

  Args:
      target_image (Image): целевое изображение (матрица цветов).
      source_images (list[Image]): список исходных изображений для сравнения с целевым.
      delta (float, optional): пороговое значение для `cv2.matchShapes`. Defaults to 0.15.
      need_to_draw_contour: (bool): флаг, отвечающий за то, нужно ли выводить изображения.

  Returns:
      list[bool]: список булевых значений, где True означает, что контур наибольшей площади
                  в соответствующем исходном изображении похож на контур наибольшей площади
                  в целевом изображении (в пределах `delta`).
  """

  target_contours = GetImageContours(target_image)

  # контур с наибольшей площадью в целевом изображении
  max_area_target_contour = max(target_contours, key=cv2.contourArea)

  if need_to_draw_contour:
    cv2.drawContours(target_image, [max_area_target_contour], -1, [0, 255, 0], 3)

  result: list[bool] = []

  for index, image in enumerate(source_images):
    # контуры из текущего исходного изображения
    contours = GetImageContours(image)

    # контур с наибольшей площадью в текущем исходном изображении
    curr_contour = max(contours, key=cv2.contourArea)

    # сравниваем контуры с использованием cv2.matchShapes и добавляем результат в список
    result.append(True if cv2.matchShapes(max_area_target_contour,
                                          curr_contour,
                  cv2.CONTOURS_MATCH_I1, 0) < delta else False)

  return result


def HOGCrossCorrelation(image_hog: Image,
                        template_hog: Image) -> Image:
  """
  Вычисляет кросс-корреляцию между HOG-визуализацией изображения и HOG-визуализацией шаблона.

  Args:
      image_hog (Image): HOG-визуализация изображения.
      template_hog (Image): HOG-визуализация шаблона, который ищем в изображении.

  Returns:
      Image: массив NumPy, представляющий карту кросс-корреляции.  
             Яркие области на карте указывают на высокую степень 
             соответствия между шаблоном и изображением.
  """

  # паддинг нужен для того, чтобы учесть все возможные позиции шаблона в изображении
  pad_rows, pad_cols = template_hog.shape[0] - 1, template_hog.shape[1] - 1

  # для избежания граничных эффектов при вычислении кросс-корреляции:
  padded_image_hog = np.pad(
      image_hog, ((pad_rows, pad_rows), (pad_cols, pad_cols)),
      mode="constant"  # Заполняем нулями
  )

  correlation = cv2.filter2D(padded_image_hog, -1, template_hog)

  return correlation


def HOG(image: Image) -> Image:
  """
  Вычисляет HOG (Histogram of Oriented Gradients) для входного изображения 
  и возвращает его визуализацию.

  Args:
      image (Image): исходное изображение (матрица цветов).

  Returns:
      Image: HOG-визуализация изображения (NumPy array).
             Представляет собой изображение, показывающее градиенты и их ориентации.
  """

  # Вычисляем HOG с использованием skimage.feature.hog.  Параметр visualize=True
  # указывает, что нам нужна визуализация HOG.
  _, hog_image = skimage.feature.hog(image, orientations=9,
                                     pixels_per_cell=(20, 20),
                                     cells_per_block=(1, 1),
                                     visualize=True,
                                     channel_axis=-1)
  return hog_image


def LocateEyes(image: Image,
               template: Image) -> None:
  """
  Находит положение глаз на изображении, используя кросс-корреляцию HOG-дескрипторов.

  Args:
      image (Image): изображение, на котором нужно найти глаза (матрица цветов).
      template (Image): шаблон глаза (матрица цветов).
  """

  image_hog = HOG(image)
  template_hog = HOG(template)

  correlation = HOGCrossCorrelation(image_hog, template_hog)

  # положение максимального значения кросс-корреляции (первый глаз)
  _, _, _, max_loc = cv2.minMaxLoc(correlation)
  pad_rows, pad_cols = template.shape[0] - 1, template.shape[1] - 1

  # координаты верхнего левого угла найденного глаза на исходном изображении
  top_left = (max_loc[0] - pad_cols,
              max_loc[1] - pad_rows)

  y, x = top_left[1], top_left[0]

  polt.figure(figsize=(10, 5))
  polt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  polt.axis("off")

  polt.plot(x, y, 'o', ms=25, mew=5, mec='k', mfc='none')

  # стираем прямоугольный столб по найденному глазу на карте корреляции,
  # чтобы лучше найти второй глаз
  correlation[:, max_loc[0] - template.shape[1] // 2: max_loc[0] + template.shape[1] // 2] = 0

  # положение максимального значения кросс-корреляции (второй глаз)
  _, _, _, max_loc = cv2.minMaxLoc(correlation)
  pad_rows, pad_cols = template.shape[0] - 1, template.shape[1] - 1
  top_left = (max_loc[0] - pad_cols, max_loc[1] - pad_rows)

  y, x = top_left[1], top_left[0]

  polt.plot(x, y, 'o', ms=25, mew=5, mec='k', mfc='none')

  polt.show()
