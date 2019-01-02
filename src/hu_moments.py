import numpy as np
import cv2


def central_moments(img_binary, order):
    # img_binary: object/foreground = 0 (chu trang tren nen den)

    # centroid
    moments = cv2.moments(img_binary)
    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]

    # central moments
    [y, x] = np.where(img_binary)
    cent_mmt = np.zeros((order + 1, order + 1))
    norm_cent_mmt = np.zeros((order + 1, order + 1))

    x_cent_x = x - centroid_x
    y_cent_y = y - centroid_y

    for i in range(0, order + 1):
        for j in range(0, order + 1 - i):
            cent_mmt[i, j] = np.sum(x_cent_x ** i * y_cent_y ** j)

            # norm central moments
            gamma = 1 + (i + j) / 2
            norm_cent_mmt[i, j] = cent_mmt[i, j] / (cent_mmt[0, 0] ** gamma)

    return norm_cent_mmt


def cal_hu_moments(img_binary):
    n = central_moments(img_binary, 3)

    n_30_add_12 = n[3, 0] + n[1, 2]
    n_30_sub_12 = n[3, 0] - 3 * n[1, 2]
    n_21_add_03 = n[2, 1] + n[0, 3]
    n_21_sub_03 = 3 * n[2, 1] - n[0, 3]
    n_20_sub_02 = n[2, 0] - n[0, 2]

    hu_2 = n_20_sub_02 ** 2 + 4 * n[1, 1] ** 2
    hu_3 = n_30_sub_12 ** 2 + n_21_sub_03 ** 2
    hu_4 = n_30_add_12 ** 2 + n_21_add_03 ** 2
    hu_5 = n_30_sub_12 * n_30_add_12 * (n_30_add_12 ** 2 - 3 * n_21_add_03 ** 2) + \
           n_21_sub_03 * n_21_add_03 * (3 * n_30_add_12 ** 2 - n_21_add_03 ** 2)
    hu_6 = n_20_sub_02 * (n_30_add_12 ** 2 - n_21_add_03 ** 2) + 4 * n[1, 1] * n_30_add_12 * n_21_add_03
    hu_7 = n_21_sub_03 * n_30_add_12 * (n_30_add_12 ** 2 - 3 * n_21_add_03 ** 2) - \
           n_30_sub_12 * n_21_add_03 * (3 * n_30_add_12 ** 2 - n_21_add_03 ** 2)

    return hu_2, hu_3, hu_4, hu_5, hu_6, hu_7
