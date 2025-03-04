import numpy as np


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
    Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image (np.ndarray): numpy array of shape (H, W).
        pad_width (int): width of the zero padding (left and right padding).
        pad_height (int): height of the zero padding (bottom and top padding).

    Returns:
        np.ndarray: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    h, w = image.shape
    h_out, w_out = (h+pad_height*2, w+pad_width*2)
    out = np.zeros((h_out, w_out))

    for i in range(pad_height, pad_height+h):
        for j in range(pad_width, pad_width+w):
            out[i, j] = image[i-pad_height, j-pad_width]

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

    prim_out = np.fft.ifft2(f_out)

    out = np.real(prim_out) - np.imag(prim_out)
    # out = (np.real(prim_out))

    return out


def cross_correlation(f, g):
    """
    Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """
    Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """
    Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return out
