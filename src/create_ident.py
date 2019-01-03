from skimage import morphology
import numpy as np
import cv2


def fn_create_ident(char):
    k = 8
    identifier = np.zeros(2 * k + 6)
    char_inv = np.ones(char.shape) - char

    pad = 3

    if sum(char_inv[0, :]) != 0:
        char = np.vstack((np.ones((pad, char.shape[1]), dtype=bool), char))
    if sum(char_inv[-1, :]) != 0:
        char = np.vstack((char, np.ones((pad, char.shape[1]), dtype=bool)))
    if sum(char_inv[:, 0]) != 0:
        char = np.hstack((np.ones((char.shape[0], pad), dtype=bool), char))
    if sum(char_inv[:, -1]) != 0:
        char = np.hstack((char, np.ones((char.shape[0], pad), dtype=bool)))

    char_inv = np.ones(char.shape) - char

    n = np.sum(char_inv)
    [y, x] = np.where(char_inv)

    char_moment = cv2.moments(char_inv.astype(float))
    cent_x = char_moment["m10"] / char_moment["m00"]
    cent_y = char_moment["m01"] / char_moment["m00"]

    inertia = sum((x - cent_x) ** 2 + (y - cent_y) ** 2)
    inertia_n2 = inertia / (n ** 2)
    identifier[0] = inertia_n2

    [y, x] = char.shape
    dr = max(x - cent_x - 1, cent_x + 1, y - cent_y - 1, cent_y - 1) / (k + 1)
    [c, r] = np.meshgrid(np.arange(0, x), np.arange(0, y))

    for i in range(k):
        rad = dr * (i + 1)

        ele_1 = np.sqrt((c - cent_x) ** 2 + (r - cent_y) ** 2) <= rad
        ele_2 = np.sqrt((c - cent_x) ** 2 + (r - cent_y) ** 2) <= (rad - 1)

        circle = np.logical_xor(ele_1, ele_2)

        cidx_idx = np.where(circle)

        if len(cidx_idx) != 0:
            c_cidx = c[cidx_idx[0], cidx_idx[1]]
            r_cidx = r[cidx_idx[0], cidx_idx[1]]

            vals = np.vstack((cidx_idx[0], cidx_idx[1], c_cidx - cent_x, r_cidx - cent_y))

            [_, theta] = cart2pol(vals[2], vals[3])
            vals = np.vstack((vals, theta))

            order = vals[4].argsort()
            sortedvals = vals[:, order]

            circ_vec = char[sortedvals[0].astype(int), sortedvals[1].astype(int)]

            circ_vec = morphology.binary_closing(circ_vec, np.ones(2))
            circ_vec = morphology.binary_opening(circ_vec, np.ones(2))

            if sum(circ_vec) != 0:
                cnt = str_find(np.hstack((np.ones(3), circ_vec)), np.array([0, 0]))
                count = sum(np.diff(np.hstack((1, cnt))) != 1)

                identifier[i + 1] = count

                circ = len(circ_vec)

                if circ_vec[0] == 1 and circ_vec[-1] == 1:
                    idx = (~circ_vec).nonzero()[0]
                    if len(idx) != 0:
                        idx = idx[-1]
                        circ_vec = np.hstack((circ_vec[idx + 1:], circ_vec[0:idx+1]))

                B = np.hstack((0, circ_vec, 0))
                bgrd_len = np.array(np.where(np.diff(B) == -1)) - np.array(np.where(np.diff(B) == 1))

                if count == 0:
                    d2 = 0
                    d1 = 0
                else:
                    d2 = np.max(bgrd_len)
                    idx_d2 = np.argmax(bgrd_len)
                    bgrd_len[0, idx_d2] = 0
                    d1 = np.max(bgrd_len)

                ratio = (d2 - d1) / circ

                if i > 0:
                    identifier[i + k] = ratio

    char_hu_moment = cv2.HuMoments(char_moment).squeeze()
    identifier[2*k:] = char_hu_moment[1:]

    return identifier


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def str_find(a, b):
    temp = rolling_window(a, len(b))
    result = np.where(np.all(temp == b, axis=1))
    return result[0]
