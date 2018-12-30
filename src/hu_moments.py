import numpy as np
import cv2


def cal_hm(img_binary):
    img_moment = cv2.moments(img_binary)

    gamma_11 = 1 + (1 + 1) / 2
    n_11 = img_moment["m11"] / (img_moment["m00"] ** gamma_11)

    gamma_20 = 1 + (2 + 0) / 2
    gamma_02 = gamma_20
    n_20 = img_moment["m20"] / (img_moment["m00"] ** gamma_20)
    n_02 = img_moment["m02"] / (img_moment["m00"] ** gamma_02)

    hu_2 = (n_20 - n_02) ** 2 + 4 * n_11 * n_11
    hu_1 = n_20 + n_02
    return hu_1
