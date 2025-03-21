import random
import sys
from typing import Any, Literal, TypedDict

import cv2
import numpy as np
import scipy as sc

from scipy.sparse import lil_matrix, linalg


from matplotlib.axes import Axes
import matplotlib.pyplot as polt  # дань "уважения" ""легенде""
from matplotlib.ticker import AutoMinorLocator


Hist = np.ndarray
Image = np.ndarray


class RGBHists(TypedDict):
  r: np.ndarray
  g: np.ndarray
  b: np.ndarray


def GetIlluminationImageHist(image: Image) -> Hist:
  """
  Вычисляет гистограмму изображения в оттенках серого, 
  которая может быть использована для анализа освещенности.

  Args:
      image (Image): входное BGR изображение.

  Returns:
      Hist: гистограмма изображения в оттенках серого. 
            Представляет собой `numpy.ndarray`, где каждый элемент соответствует
            количеству пикселей с определенным значением яркости (от `0` до `255`).
  """

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return cv2.calcHist(images=[gray_image],
                      channels=[0],    # для оттенков серого: [0] (только один канал)
                      mask=None,
                      histSize=[256],  # гистограмма с 256 bins для одного канала
                      ranges=[0, 256]  # типичный диапазон для изображений с типом uint8
                      )


def GetRGBImageHists(image: Image) -> RGBHists:
  """
  Вычисляет гистограммы для каждого цветового канала (красный, зеленый, синий) RGB-изображения.

  Args:
      image (Image): входное BGR изображение. 

  Returns:
      RGBHists: словарь, содержащий гистограммы для красного ('r'), 
                зеленого ('g') и синего ('b') каналов.
                Каждая гистограмма представляет собой numpy.ndarray, 
                где каждый элемент соответствует количеству пикселей 
                с определенным значением яркости в соответствующем цветовом канале (от 0 до 255).
  """

  RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  return {
      "r": cv2.calcHist([RGB_image], [0], None, [256], [0, 256]),
      "g": cv2.calcHist([RGB_image], [1], None, [256], [0, 256]),
      "b": cv2.calcHist([RGB_image], [2], None, [256], [0, 256]),
  }


def IsRGBHists(to_plot: dict) -> bool:
  """
  Проверяет, является ли входной словарь гистограммами RGB-изображения.

  Args:
      to_plot (dict): словарь, который необходимо проверить.

  Returns:
      bool: True, если словарь соответствует структуре RGBHists, иначе False.
  """

  return (type(to_plot) == RGBHists) or (to_plot.keys() == {"r", "g", "b"})


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


def CreateSparseMatrix(source: Image,
                       target: Image,
                       mask: Image,
                       alpha: float) -> tuple[lil_matrix, np.ndarray]:
  """
  Создает разреженную матрицу и массив, используя входные изображения и параметры.

  Args:
      source (Image): исходное изображение, из которого будет создана разреженная матрица
                      (будет использоваться в качестве эталона для 
                      создания разреженной матрицы)
      target (Image): изображение, представляющее целевое изображение, к которому будет 
                      применена разреженная матрица. 
                      (используется в качестве ссылки для создания разреженной 
                      матрицы на основе предоставленного исходного изображения и маски)
      mask (Image): изображение, определяющее области интереса или регионы, которые будут 
                    использоваться для создания разреженной матрицы. 
      alpha (float): коэффициент смешивания, определяющий вклад градиента 
                     исходного изображения в результирующее изображение.

  Returns:
      tuple[lil_matrix, np.ndarray]: кортеж, содержащий разреженную матрицу (A) и массив (b), 
                                     используемые для решения системы уравнений A*x = b.
  """

  def CalcSparseMatrixElement(A: lil_matrix,
                              b: np.ndarray,
                              row: int,
                              column: int):
    """
    Вычисляет элементы разреженной матрицы и вектора b для конкретного пикселя.

    Args:
        A (lil_matrix): разреженная матрица, в которую будут добавлены коэффициенты.
        b (np.ndarray): массив, содержащий целевые значения градиента для каждого пикселя.
        row (int): строка пикселя.
        column (int): столбец пикселя.
    """

    if mask[row, column] > 0:
      src_grad = ComputeGradient(source, row, column)
      trg_grad = ComputeGradient(target, row, column)

      # gradient = (alpha)*source_gradient + (1 - alpha)*target_gradient
      b[index] = alpha * src_grad + (1 - alpha) * trg_grad

      A[index, index] = 4

      for direction in {(-1, 0), (0, -1), (1, 0), (0, 1)}:
        dr, dc = direction
        new_row, new_col = row + dr, column + dc

        if -1 < new_row < height and -1 < new_col < width:
          neighbor_idx = new_row * width + new_col
          A[index, neighbor_idx] = -1
    else:
      # копируем пиксель из целевого изображения
      A[index, index] = 1
      b[index] = target[row, column]

  height, width = target.shape
  num_pixels = height * width

  # A*x = b, где
  # A — матрица разреженных коэффициентов
  # x — выходное изображение (в виде столбца)
  # b — желаемая матрица градиента

  A = lil_matrix((num_pixels, num_pixels))
  b = np.zeros(num_pixels)

  for row in range(height):
    for col in range(width):
      index = row * width + col

      CalcSparseMatrixElement(A, b, row, col)

  return A, b


def ComputeGradient(image: Image,
                    row: int,
                    column: int) -> float:
  """
  Вычисляет градиент пикселя на основе разницы с его соседями.

  Args:
      image (Image): изображение, для которого вычисляется градиент.
      row (int): строка пикселя.
      column (int): столбец пикселя.

  Returns:
      float: вычисленное значение градиента.
  """

  # 4*x(row, col) - x(row+1, col) - x(row-1, col) - x(row, col+1)
  # - x(row, col-1) = desired pixel gradient
  height, width = image.shape
  gradient = 4 * image[row, column]

  for direction in {(-1, 0), (0, -1), (1, 0), (0, 1)}:
    dr, dc = direction

    if -1 < row + dr < height and -1 < column + dc < width:
      gradient -= image[row + dr, column + dc]

  return gradient


def BlendedImage(source: Image,
                 target: Image,
                 mask: Image,
                 alpha: float) -> Image:
  """
  Выполняет смешивание изображений по каналам.

  Args:
      source (Image): исходное изображение.
      target (Image): целевое изображение.
      mask (Image): маска, определяющая области смешивания.
      alpha (float): коэффициент смешивания.

  Returns:
      Image: смешанное изображение.
  """

  blended_image = np.zeros_like(target)

  for channel in range(3):
    A, b = CreateSparseMatrix(source[:, :, channel],
                              target[:, :, channel],
                              mask, alpha)

    blended_channel = linalg.lsqr(A, b)[0]

    blended_channel[blended_channel > 255] = 255
    blended_channel[blended_channel < 0] = 0
    blended_channel = blended_channel.astype(np.uint8)

    # так как на выходе хотим получить изображение, а не столбец
    # (который использовался всё это время)
    blended_image[:, :, channel] = blended_channel.reshape(target.shape[:2])

  return blended_image


def CoverWithCells(image: Image,
                   cell_size: int,
                   cells_amount: int
                   ) -> tuple[Image, list[tuple[int, int]]]:
  """
  Покрывает изображение ячейками заданного размера, 
  стараясь выделить области с наибольшим количеством черных пикселей.

  Args:
      image (Image): исходное изображение (numpy ndarray). Должно быть бинарным (черно-белым).
      cell_size (int): размер стороны квадратной ячейки в пикселях.
      cells_amount (int): максимальное количество ячеек, которые нужно выделить.

  Returns:
      tuple[Image, list[tuple[int, int]]]: кортеж, содержащий:
          - celled_image (Image): Изображение, на котором выделены ячейки (белые прямоугольники).
          - cell_corners (list[tuple[int, int]]): Список координат верхних левых углов выделенных ячеек.
  """

  def CalcBlackPixelPercentage(image: Image,
                               top_left_corner: tuple[int, int]) -> int:
    """
    Вычисляет процент черных пикселей в заданной области изображения.

    Args:
        image (Image): исходное изображение (numpy ndarray).
        top_left_corner (tuple[int, int]): координаты верхнего левого угла области.

    Returns:
        int: процент черных пикселей в области (от 0 до 100). 
             Возвращает 0, если область выходит за границы изображения.
    """

    x, y = top_left_corner
    bottom_right_corner = (x + cell_size, y + cell_size)

    height, width = image.shape
    if x < 0 or y < 0 or bottom_right_corner[0] > width or bottom_right_corner[1] > height:
      return 0

    square_area = image[y:bottom_right_corner[1], x:bottom_right_corner[0]]

    black_pixels = np.sum(square_area == 0)

    percentage = (black_pixels / cell_size**2) * 100

    return percentage

  # список для хранения координат углов ячеек
  cell_corners: list[tuple[int, int]] = []
  height, _ = image.shape

  celled_image = image.copy()

  # списки для хранения информации о строках и столбцах
  lines_rows: list[int] = []
  left_right_black_pixels_cols: list[tuple[int, int]] = []

  # проходим по изображению с шагом, равным размеру ячейки (по вертикали)
  for y in range(0, height, cell_size):
    start_y = y + 1 if y + 1 < height else height
    end_y = min(y + cell_size, height)

    # индексы черных пикселей в текущей строке
    row_indices, col_indices = np.where(celled_image[start_y:end_y, :] == 0)

    # самые левые и самые правые координаты черных пикселей в текущей строке
    try:
      left_right_black_pixels_cols.append((np.min(col_indices).astype(int),
                                           np.max(col_indices).astype(int)))
    except ValueError:
      left_right_black_pixels_cols.append((0, 0))

    row_indices = row_indices + start_y

    lines_rows.append(start_y - 1)
    # рисуем горизонтальную линию в celled_image
    celled_image[start_y - 1, :] = 0

  # определяем количество строк
  len_lines_rows = len(lines_rows) if lines_rows[-1] == height else len(lines_rows) - 1

  # нужно ли выбирать только одну ячейку на строке
  is_only_one = len_lines_rows >= cells_amount

  # проходим по вычисленным строкам
  for line_index in range(len_lines_rows):
    left_b, right_b = left_right_black_pixels_cols[line_index]

    curr_pos = left_b

    # перебираем координаты по горизонтали, пока не достигнем правой границы
    while (curr_pos < right_b):
      curr_corner: tuple[int, int] = (curr_pos, lines_rows[line_index])

      is_appended: bool = False
      if CalcBlackPixelPercentage(celled_image, curr_corner) > 25 and\
              len(cell_corners) < cells_amount:
        cell_corners.append(curr_corner)

        x, y = curr_corner
        bottom_right_corner = (x + cell_size, y + cell_size)
        cv2.rectangle(celled_image, curr_corner,
                      bottom_right_corner, (255, 255, 255), -1)

        is_appended = True

        # нужно выбрать только одну ячейку на строке, выходим из цикла
        if is_only_one:
          break

      curr_pos += cell_size if is_appended else 1

  return celled_image, cell_corners


def FindAndCalcCells(image: Image,
                     cell_size: int,
                     cells_amount: int) -> list[tuple[int, int]]:
  """
  Находит и вычисляет координаты ячеек на изображении, 
  содержащих наибольшее количество черных пикселей.

  Args:
      image (Image): исходное цветное изображение (numpy ndarray).
      cell_size (int): размер стороны квадратной ячейки в пикселях.
      cells_amount (int): максимальное количество ячеек, которые нужно найти.

  Returns:
      list[tuple[int, int]]: список координат верхних левых углов найденных ячеек.
  """

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
  _, binary_image = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)

  # VerbosePlot(binary_image, is_image=True)

  celled_image, corners = CoverWithCells(binary_image, cell_size, cells_amount)

  # VerbosePlot(celled_image, is_image=True)

  return corners


def CreateMosaic(image: Image,
                 coords: list[tuple[int, int]],
                 cell_size: int,
                 cell_number: int) -> Image:
  """
  Создает мозаику из фрагментов изображения, расположенных в заданных координатах.

  Args:
      image (Image): исходное изображение (numpy ndarray).
      coords (list[tuple[int, int]]): список координат верхних левых углов фрагментов изображения, которые будут использоваться для создания мозаики.
      cell_size (int): размер стороны квадратного фрагмента изображения в пикселях.
      cell_number (int): максимальное количество фрагментов, которые будут использованы для создания мозаики.

  Returns:
      Image: изображение мозаики (numpy ndarray). 
             Возвращает пустой массив, если список координат пуст.
  """

  if not coords:
    return np.array([])

  effective_cell_number = min(cell_number, len(coords))

  num_cols = int(np.ceil(np.sqrt(effective_cell_number)))
  num_rows = int(np.ceil(effective_cell_number / num_cols))

  mosaic = np.full((num_rows * cell_size, num_cols * cell_size,
                    image.shape[2]), 255, dtype=image.dtype)

  for i in range(effective_cell_number):
    x, y = coords[i]

    if y + cell_size > image.shape[0] or x + cell_size > image.shape[1]:
      print(f"CreateMosaic: coordinate ({x}, {y}) leads to out-of-bounds access, skip.")
      continue

    row = i // num_cols
    col = i % num_cols
    mosaic[row * cell_size:(row + 1) * cell_size,
           col * cell_size:(col + 1) * cell_size] =\
        image[y:y + cell_size, x:x + cell_size]

  return mosaic


def DrawCellBoundaries(image: Image,
                       coords: list[tuple[int, int]],
                       cell_size: int,
                       color: tuple[int, int, int] | int = (0, 255, 0),
                       thickness: int = 4) -> Image:
  """
  Рисует границы вокруг ячеек на изображении по заданным координатам.

  Args:
      image (Image): исходное изображение (numpy ndarray).
      coords (list[tuple[int, int]]): список координат верхних левых углов ячеек.
      cell_size (int): размер стороны квадратной ячейки в пикселях.
      color (tuple[int, int, int] | int, optional): цвет границы ячейки. Может быть кортежем (R, G, B) или целым числом (для оттенков серого). Defaults to (0, 255, 0).
      thickness (int, optional): толщина границы ячейки в пикселях. Defaults to 4.

  Returns:
      Image: изображение с нарисованными границами ячеек (numpy ndarray). 
                  Возвращает копию исходного изображения, если список координат пуст.
  """

  if not coords:
    return image.copy()

  output_image = image.copy()

  for x, y in coords:
    if y + cell_size > output_image.shape[0] or\
            x + cell_size > output_image.shape[1]:
      print(
        f"DrawCellBoundaries: coordinate ({x}, {y}) leads to out-of-bounds access, skip.")
      continue

    cv2.rectangle(output_image, (x, y), (x + cell_size, y + cell_size), color, thickness)

  return output_image
