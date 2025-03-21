from typing import Callable
import numpy as np

import enum


class ConvLevel(enum.Enum):
  Nested = 0
  Fast = 1
  Faster = 2
  Best = 3


def IsNonNegative(number: int | float):
  return number >= 0


def ConvFilterNested(image: np.ndarray,
                     kernel: np.ndarray) -> np.ndarray:
  """
  Наивная реализация фильтра свертки.

  Это наивная реализация свертки с использованием 4 вложенных циклов for.
  Эта функция вычисляет свертку изображения с ядром и выводит
  результат, который имеет ту же форму, что и входное изображение.

  Args:
      image (np.ndarray): изображение (матрица).
      kernel (np.ndarray): матрица ядра.

  Returns:
      np.ndarray: изображение с примененным ядром.
  """

  h, w = image.shape
  h_kernel, w_kernel = kernel.shape
  out: np.ndarray = np.zeros(image.shape)

  pad_height = h_kernel // 2
  pad_width = w_kernel // 2

  for i in range(h):
    for j in range(w):
      sum_kernel: int = 0

      for i_kernel in range(-pad_height, pad_height + 1):
        for j_kernel in range(-pad_width, pad_width + 1):
          if IsNonNegative(i + i_kernel) and IsNonNegative(j + j_kernel) and\
                  i + i_kernel < h and j + j_kernel < w:
            sum_kernel += (
                image[i + i_kernel, j + j_kernel]
                * kernel[(h_kernel - 1) // 2 - i_kernel,
                         (w_kernel - 1) // 2 - j_kernel]
            )

      out[i, j] = sum_kernel

  return out


def ZeroPad(image: np.ndarray, pad_height: int, pad_width: int):
  """
  Заполняет изображение нулями по паддингу.

  Example: a 1x1 image [[1]], pad_height = 1, pad_width = 2:

      [[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]]         of shape (3, 5)

  Args:
      image (np.ndarray): изображение (матрица).
      pad_width (int): ширина нулевого отступа (левый и правый отступ).
      pad_height (int): высота нулевого отступа (нижнего и верхнего отступа).

  Returns:
      np.ndarray: изображение с отступами (матрица: [H+2*pad_height, W+2*pad_width]).
  """

  h, w = image.shape
  h_out, w_out = (h + pad_height * 2, w + pad_width * 2)
  out = np.zeros((h_out, w_out))

  for i in range(pad_height, pad_height + h):
    for j in range(pad_width, pad_width + w):
      out[i, j] = image[i - pad_height, j - pad_width]

  return out


def ConvFilterFast(image: np.ndarray,
                   kernel: np.ndarray) -> np.ndarray:
  """ 
  Эффективная реализация фильтра свертки.

  Эта функция использует поэлементное умножение и np.sum()
  для эффективного вычисления взвешенной суммы соседства в каждом
  пикселе.

  Подсказки:
  - Используйте функцию zero_pad, которую вы реализовали выше
  - Должно быть два вложенных цикла for
  - Вам могут пригодиться np.flip() и np.sum()

  Args:
      image (np.ndarray): изображение (матрица).
      kernel (np.ndarray): матрица ядра.

  Returns:
      np.ndarray: изображение с примененным ядром.
  """

  h, w = image.shape
  h_kernel, w_kernel = kernel.shape
  out = np.zeros(image.shape)

  pad_height = h_kernel // 2
  pad_width = w_kernel // 2

  padded_image = ZeroPad(image, pad_height, pad_width)

  for i in range(h):
    for j in range(w):
      out[i, j] = np.sum(padded_image[i:i + h_kernel, j:j + w_kernel] * np.flip(kernel))

  return out


def ConvFilterFaster(image: np.ndarray,
                     kernel: np.ndarray) -> np.ndarray:
  """
  Args:
      image (np.ndarray): изображение (матрица).
      kernel (np.ndarray): матрица ядра.

  Returns:
      np.ndarray: изображение с примененным ядром.
  """

  out = np.zeros(image.shape)

  f_image = np.fft.fft2(image)
  f_kernel = np.fft.fft2(kernel, image.shape[:2])

  f_out = f_image * f_kernel

  # prim_out = np.fft.ifft2(f_out)
  prim_out = np.flip(np.fft.fft2(f_out))  # почему-то это работает лучше

  return np.real(prim_out)  # - np.imag(prim_out)


def ConvFilterBest(image: np.ndarray,
                   kernel: np.ndarray) -> np.ndarray:
  """
  Args:
      image (np.ndarray): изображение (матрица).
      kernel (np.ndarray): матрица ядра.

  Returns:
      np.ndarray: изображение с примененным ядром.
  """
  def CircularExtension2d(kernel: np.ndarray,
                          num_rows: int,
                          num_cols: int):
    """ Циклическое расширение для ядра из ConvFilterBest """

    kernel_radius_v = kernel.shape[0] // 2  # Вертикальный радиус
    kernel_radius_h = kernel.shape[1] // 2  # Горизонтальный радиус

    kernel_padded = np.zeros((num_rows, num_cols), dtype=kernel.dtype)
    kernel_padded[: kernel.shape[0], : kernel.shape[1]] = kernel

    kernel_padded = np.roll(
        kernel_padded, shift=(-kernel_radius_v, -kernel_radius_h), axis=(0, 1)
    )

    return kernel_padded

  h, w = image.shape

  padded_image = np.zeros([h + w - 1, h + w - 1])
  padded_image[:h, :w] = image

  padded_kernel = CircularExtension2d(
      kernel, padded_image.shape[0], padded_image.shape[1]
  )

  f_image = np.fft.fft2(padded_image)
  f_kernel = np.fft.fft2(padded_kernel)

  f_out = f_image * f_kernel

  prim_out = np.fft.ifft2(f_out)

  return np.real(prim_out)[:h, :w]


# MARK: conv_dict
conv_dict: dict[ConvLevel, Callable] = {
    ConvLevel.Nested: ConvFilterNested,
    ConvLevel.Fast: ConvFilterFast,
    ConvLevel.Faster: ConvFilterFaster,
    ConvLevel.Best: ConvFilterBest,
}


def CrossCorrelation(f: np.ndarray,
                     g: np.ndarray,
                     conv_level: ConvLevel = ConvLevel.Best) -> np.ndarray:
  """
  Кросс-корреляция f и g.

  Args:
      f (np.ndarray).
      g (np.ndarray).
      conv_type (conv_type): уровень корреляции.

  Returns:
      np.ndarray: результат кросс-корреляции.
  """

  return conv_dict.get(conv_level, ConvFilterBest)(f, np.flip(g))


def ZeroMeanCrossCorrelation(f: np.ndarray,
                             g: np.ndarray,
                             conv_level: ConvLevel = ConvLevel.Best) -> np.ndarray:
  """
  Нулевая средняя кросс-корреляция f и g.

  Args:
      f (np.ndarray).
      g (np.ndarray).
      conv_type (conv_type): уровень корреляции.

  Returns:
      np.ndarray: результат кросс-корреляции.
  """

  return conv_dict.get(conv_level, ConvFilterBest)(f, np.flip(g - np.mean(g)))


def NormalizedCrossCorrelation(f: np.ndarray,
                               g: np.ndarray) -> np.ndarray:
  """
  Нормализованная кросс-корреляция f и g.

  Нормализация подизображения f и шаблона g на каждом шаге
  перед вычислением взвешенной суммы обоих.

  Args:
      f (np.ndarray).
      g (np.ndarray).

  Returns:
      np.ndarray: результат кросс-корреляции.
  """

  h_f, w_f = f.shape
  h_g, w_g = g.shape

  out = np.zeros((h_f, w_f))
  pad_f = ZeroPad(f, h_g // 2, w_g // 2)

  g_normalized = (g - np.mean(g)) / np.std(g)

  for i_f in range(h_f):
    for j_f in range(w_f):
      sub_f = pad_f[i_f: i_f + h_g, j_f: j_f + w_g]
      sub_f = (sub_f - np.mean(sub_f)) / np.std(sub_f)

      out[i_f, j_f] = np.sum(sub_f * g_normalized)

  return out


def IsProductOnShelf(shelf: np.ndarray,
                     product: np.ndarray,
                     conv_level: ConvLevel = ConvLevel.Best) -> None:
  out = ZeroMeanCrossCorrelation(shelf, product, conv_level)

  out = out / float(product.shape[0] * product.shape[1])

  out = out > 1500.0

  if np.sum(out) > 0:
    print(f'{conv_level}: the product is on the shelf')
  else:
    print(f'{conv_level}: the product is not on the shelf')
