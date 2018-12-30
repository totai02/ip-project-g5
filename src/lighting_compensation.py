from skimage import morphology
from scipy import ndimage, signal
import numpy as np
import cv2


def fn_lighting_compensation(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    [height, width, _] = img.shape
    num_mid_gray = np.sum(np.sum((gray_image < 240) & (gray_image > 15)))
    if num_mid_gray < 0.1 * height * width:
        [_, bw_img] = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        win_size = round(min(height / 60, width / 60))

        window_means = signal.fftconvolve(gray_image, np.ones((win_size, win_size)) * (1 / win_size ** 2), mode='same')
        demeaned = window_means.astype(int) - gray_image.astype(int) - 10
        demeaned[demeaned < 0] = 0
        demeaned = demeaned.astype(np.uint8)

        [_, bw_img] = cv2.threshold(demeaned, 0, 255, cv2.THRESH_BINARY)
        bw_img = bw_img.astype(bool)
        noise_size = round(0.0001 * height * width)

        bw_img = morphology.remove_small_objects(bw_img, min_size=noise_size)
        bw_img = ndimage.binary_closing(bw_img, structure=np.ones((4, 4)))

        small_hole_thresh = round(0.0001 * height * width)
        filled = ndimage.binary_fill_holes(bw_img)
        holes = filled & ~bw_img
        lg_holes = morphology.remove_small_objects(holes, min_size=small_hole_thresh)
        sm_holes = holes & ~lg_holes
        bw_img = bw_img | sm_holes
        bw_img = bw_img.astype(np.uint8) * 255

    bw_img = ~bw_img

    return bw_img


def fspecial(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
