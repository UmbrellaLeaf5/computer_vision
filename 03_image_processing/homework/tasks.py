import sys
from typing import Any, TypedDict

import numpy as np
import cv2

from matplotlib.axes import Axes
import matplotlib.pyplot as polt  # дань "уважения" ""легенде""
from matplotlib.ticker import AutoMinorLocator


Hist = np.ndarray


class RGBHists(TypedDict):
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray


def GetIlluminationImageHist(image: np.ndarray) -> Hist:
    """
    Вычисляет гистограмму изображения в оттенках серого, 
    которая может быть использована для анализа освещенности.

    Args:
        image (np.ndarray): входное BGR изображение.

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


def GetRGBImageHists(image: np.ndarray) -> RGBHists:
    """
    Вычисляет гистограммы для каждого цветового канала (красный, зеленый, синий) RGB-изображения.

    Args:
        image (np.ndarray): входное BGR изображение. 

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
                figure_size: tuple[float, float] = (10, 6)):
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
        axs.plot(to_plot)

    SetGrid(axs)
