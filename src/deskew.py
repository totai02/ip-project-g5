from skimage import feature, transform, morphology
from scipy import stats, misc
import numpy as np


def fn_deskew(bw_img, soften_flag=False):
    edges = feature.canny(bw_img)
    h, theta, d = transform.hough_line(edges)

    P = transform.hough_line_peaks(h, theta, d, num_peaks=4)

    orientations = P[1]

    if len(np.unique(orientations)) == 4:
        dominant_orientation = orientations[0]
    else:
        dominant_orientation = stats.mode(orientations).mode[0]

    deskewing_angle = dominant_orientation - np.pi / 2

    while abs(deskewing_angle) > np.pi / 4:
        deskewing_angle = -deskewing_angle + np.sign(deskewing_angle) * np.pi / 2

    deskew_img = misc.imrotate((bw_img == 0).astype(float), np.rad2deg(deskewing_angle), interp="nearest")
    deskew_img = (deskew_img == 0).astype(float)

    if soften_flag and abs(deskewing_angle / np.pi * 180) > 5:
        deskew_img = fn_soften_edges(deskew_img, 5)

    return deskew_img


def fn_soften_edges(bw_img, se_size):
    se = morphology.square(se_size)
    softened = morphology.opening(bw_img, se)
    return softened

