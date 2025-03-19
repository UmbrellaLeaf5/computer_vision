from typing import Any
from matplotlib.axes import Axes
import numpy as np
import cv2
import matplotlib.pyplot as polt
from matplotlib.ticker import AutoMinorLocator

Hist = np.ndarray
RGBHists = dict[str, np.ndarray]


def GetIlluminationImageHist(image_path: str) -> Hist:
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

    return cv2.calcHist(images=[gray_image.ravel()],
                        channels=[0],    # для оттенков серого: [0] (только один канал)
                        mask=None,
                        histSize=[256],  # гистограмма с 256 bins для одного канала
                        ranges=[0, 256]  # типичный диапазон для изображений с типом uint8
                        )


def GetRGBImageHists(image_path: str) -> RGBHists:
    image = cv2.imread(image_path)

    RGB_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    return {
        "r": cv2.calcHist([RGB_image], [0], None, [256], [0, 256]),
        "g": cv2.calcHist([RGB_image], [1], None, [256], [0, 256]),
        "b": cv2.calcHist([RGB_image], [2], None, [256], [0, 256]),
    }


def IsRGBHists(to_plot: dict) -> bool:
    colors = ("r", "g", "b")
    return (type(to_plot) == RGBHists) or\
        ((type(to_plot) == dict) and (all(key in to_plot for key in colors)))


def PlotImagesRGBHists(image_paths: list[str],
                       title: str = "",
                       hists_titles: list[str] = []):
    amount = len(image_paths)

    if amount != len(hists_titles) and hists_titles != []:
        raise ValueError(
            "PlotImagesRGBHists: lengths of `image_paths` and `hists_titles` "
            f"do not match ({amount} and {len(hists_titles)}).")

    n_rows = amount // 3 if amount % 3 == 0 else amount // 3 + 1
    fig, axs = polt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    fig.suptitle(title, fontsize=24)

    for i, ax in enumerate(axs.flatten()):
        if i < amount:
            to_plot = GetRGBImageHists(image_paths[i])

            for color in ("r", "g", "b"):
                ax.plot(to_plot[color],
                        color=color,
                        label=color.upper())

                ax.legend()

            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Frequency")

            ax.set_title(
                f"{image_paths[i] if hists_titles == [] else hists_titles[i]}" if
                i < amount else "")

            SetGrid(ax)

    fig.tight_layout()


def PlotImages(image_paths: list[str],
               title: str = "",
               hists_titles: list[str] = []):
    amount = len(image_paths)

    if amount != len(hists_titles) and hists_titles != []:
        raise ValueError(
            "PlotImages: lengths of `image_paths` and `hists_titles` "
            f"do not match ({amount} and {len(hists_titles)}).")

    n_rows = amount // 3 if amount % 3 == 0 else amount // 3 + 1
    fig, axs = polt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    fig.suptitle(title, fontsize=24)

    for i, ax in enumerate(axs.flatten()):
        if i < amount:
            image = cv2.imread(image_paths[i])
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

            ax.imshow(image)
            ax.axis('off')

            ax.set_title(
                f"{image_paths[i] if hists_titles == [] else hists_titles[i]}" if
                i < amount else "")

            SetGrid(ax)

    fig.tight_layout()


def SetGrid(ax: Axes | None = None,
            n_locator: int = 10,
            minor_line_width: float = 0.2,
            major_line_width: float = 0.4):
    if ax is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_locator))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_locator))

    polt.grid(which='minor', linestyle='--', linewidth=minor_line_width)
    polt.grid(which='major', linewidth=major_line_width)


def VerbosePlot(to_plot: Any,
                title: str = "",
                x_label: str = "",
                y_label: str = ""):
    _, axs = polt.subplots(1, 1, figsize=(10, 6))

    axs.set_title(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)

    if type(to_plot) == dict:
        colors = ("r", "g", "b")

        if IsRGBHists(to_plot):
            for color in colors:
                axs.plot(to_plot[color],
                         color=color,
                         label=color.upper())

        else:
            for key in to_plot.keys():
                axs.plot(to_plot[key],
                         label=key)

        polt.legend()
    else:
        axs.plot(to_plot)

    SetGrid(axs)


def ForestOrDesert(image_path: str) -> str:
    hists = GetRGBImageHists(image_path)

    return "Desert" if np.argmax(hists['r']) > np.argmax(hists['g']) else "Forest"
