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

from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure

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
  _summary_

  Args:
      image (Image): _description_

  Returns:
      list[Contour]: _description_
  """

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(gray)

  blur = cv2.medianBlur(clahe, 9)
  dilate = cv2.dilate(blur, np.ones((5, 5), np.uint8), iterations=3)

  blur = cv2.bilateralFilter(dilate, d=9, sigmaColor=50, sigmaSpace=40)
  th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

  erode = cv2.dilate(th2, np.ones((3, 3), np.uint8), iterations=3)
  contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  return list(contours)


def RightImagesBoolList(target_image: Image,
                        source_images: list[Image],
                        delta=0.1) -> list[bool]:
  """
  _summary_

  Args:
      target_image (Image): _description_
      source_images (list[Image]): _description_
      delta (float, optional): _description_. Defaults to 0.1.

  Returns:
      list[bool]: _description_
  """

  target_contours = GetImageContours(target_image)

  max_area_target_contour = max(target_contours, key=cv2.contourArea)

  cv2.drawContours(target_image, [max_area_target_contour], -1, [0, 255, 0], 3)

  result: list[bool] = []

  for index, image in enumerate(source_images):
    contours = GetImageContours(image)
    curr_contour = max(contours, key=cv2.contourArea)

    result.append(True if cv2.matchShapes(max_area_target_contour,
                                          curr_contour,
                  cv2.CONTOURS_MATCH_I1, 0) < delta else False)

  return result
