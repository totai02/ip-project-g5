from skimage import morphology, filters
from scipy import signal
import numpy as np
import cv2


def fn_lighting_compensation(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    [height, width, _] = img.shape
    num_mid_gray = np.sum(np.sum((gray_image < 240) & (gray_image > 15)))
    if num_mid_gray < 0.1 * height * width:
        thresh = filters.threshold_otsu(gray_image)
        bw_img = gray_image <= thresh
    else:
        win_size = round(min(height / 60, width / 60))

        window_means = signal.fftconvolve(gray_image, np.ones((win_size, win_size)) * (1 / win_size ** 2), mode='same')
        demeaned = window_means.astype(int) - gray_image.astype(int) - 10
        bw_img = demeaned > 0

        noise_size = round(0.0001 * height * width)

        bw_img = morphology.remove_small_objects(bw_img, min_size=noise_size)
        bw_img = morphology.closing(bw_img, np.ones((5, 5)))

        small_hole_thresh = round(0.0001 * height * width)
        bw_img = morphology.remove_small_holes(bw_img, area_threshold=small_hole_thresh)

    bw_img = ~bw_img

    return bw_img

