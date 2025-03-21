import sys
from typing import Any, TypedDict

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


def PlotImages(image_paths: list[str],
               title: str = "",
               hists_titles: list[str] | None = None,
               plot_only_hists: bool = False):
    """
    Отображает изображения или их RGB гистограммы в виде сетки графиков.

    Args:
        image_paths (list[str]): список путей к изображениям для отображения.
        title (str, optional): общий заголовок для всей сетки графиков. Defaults to "".
        hists_titles (list[str] | None, optional): список заголовков для гистограмм 
                                                      (или изображений, если 
                                                      `plot_only_hists==False`).
                                                      Если None,
                                                      используются пути к изображениям. 
                                                      Defaults to None.
        plot_only_hists (bool, optional): если True, отображаются только гистограммы, 
                                          иначе - изображения. Defaults to False.

    Raises:
        ValueError: Если длины списков `image_paths` и `hists_titles` не совпадают.
    """

    amount = len(image_paths)

    if amount == 0:
        return

    if hists_titles is None:
        hists_titles = image_paths  # используем пути, заголовки не предоставлены

    if amount != len(hists_titles):
        raise ValueError(
            "PlotImages: lengths of `image_paths` and `hists_titles` "
            f"do not match ({amount} and {len(hists_titles)}).")

    n_rows = (amount + 2) // 3
    fig, axs = polt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    fig.suptitle(title, fontsize=24)

    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")

        if i < amount:
            ax.set_title(hists_titles[i])

            if not plot_only_hists:
                try:
                    image = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)

                    ax.imshow(image)

                except Exception as exception:
                    print("PlotImages: error loading or displaying image "
                          f"`{image_paths[i]}`: {exception}", file=sys.stderr)

            else:
                try:
                    hist = GetRGBImageHists(cv2.imread(image_paths[i]))

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
                          f"`{image_paths[i]}`: {exception}", file=sys.stderr)

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
        figure_size (Tuple[float, float], optional): размер фигуры (ширина, высота). 
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
    def CalcSparseMatrixElement(A: lil_matrix,
                                b: np.ndarray,
                                row: int,
                                column: int,
                                mask: Image):
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

            CalcSparseMatrixElement(A, b, row, col, mask)

    return A, b


def ComputeGradient(image: Image,
                    row: int,
                    column: int) -> float:
    # 4*x(row, col) - x(row+1, col) - x(row-1, col) - x(row, col+1)
    # - x(row, col-1) = desired pixel gradient
    height, width = image.shape
    gradient = 4 * image[row, column]

    for direction in {(-1, 0), (0, -1), (1, 0), (0, 1)}:
        dr, dc = direction

        if -1 < row + dr < height and -1 < column + dc < width:
            gradient -= image[row + dr, column + dc]

    return gradient


def BlendColorChannels(source: Image,
                       target: Image,
                       mask: Image,
                       alpha: float) -> Image:
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
