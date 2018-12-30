import numpy as np
import cv2
import hu_moments
from skimage import morphology as skimorph
import scipy.io as sio
import coordinateHelper as coor


def create_char_id(char_binary):
    char_id = np.zeros((1, 1 + 15 + 6))

    # Convert input image to binary (for testing when input is an image)
    retval, threshold_img = cv2.threshold(char_binary, 50, 255, cv2.THRESH_BINARY)
    char_binary = threshold_img / 255
    char_img_inv = np.ones(char_binary.shape) - char_binary

    # # If white border on any edge, pad with more 1s
    # pad = 3
    # # top
    # if np.sum(char_img_inv[0, :]) == 0:
    #     char_binary = np.vstack((np.ones((pad, char_binary.shape[1])), char_binary))
    # # bottom
    # if np.sum(char_img_inv[-1, :]) == 0:
    #     char_binary = np.vstack((char_binary, np.ones((pad, char_binary.shape[1]))))
    # # left
    # if np.sum(char_img_inv[:, 0]) == 0:
    #     char_binary = np.hstack((np.ones((char_binary.shape[0], pad)), char_binary))
    # # right
    # if np.sum(char_img_inv[-1, :]) == 0:
    #     char_binary = np.hstack((np.ones((char_binary.shape[0], pad)), char_binary))

    # Centroid
    char_moment = cv2.moments(char_img_inv)
    cent_x = char_moment["m10"] / char_moment["m00"]
    cent_y = char_moment["m01"] / char_moment["m00"]

    # Normalized central moment of inertia
    num_char_px = np.sum(char_img_inv)
    [y, x] = np.where(char_img_inv)
    I = np.sum((x - cent_x) ** 2 + (y - cent_y) ** 2)
    norm_central = I / (num_char_px ** 2)

    # save to identifier
    char_id[:, 0] = norm_central

    # Circular topology
    k = 8

    coding_count = {}
    coding_ratio = {}

    y = char_img_inv.shape[0] - 1
    x = char_img_inv.shape[1] - 1
    dr = np.max([x - cent_x, cent_x, y - cent_y, cent_y]) / (k + 1)
    print(dr)
    [c, r] = np.meshgrid(np.arange(0, x+1, 1), np.arange(0, y+1, 1))

    for i in range(1, k+1):
        rad = dr * i

        # create circle line of logical 1s to extract template
        ele_1 = np.sqrt((c - cent_x) ** 2 + (r - cent_y) ** 2) <= rad
        ele_2 = np.sqrt((c - cent_x) ** 2 + (r - cent_y) ** 2) <= (rad - 1)
        C = np.logical_xor(ele_1, ele_2)

        # extract circular template linear indices (sort based on theta)
        [cidx_row, cidx_col] = np.where(C)

        if cidx_row != []:
            cidx_idx = np.where(C)

            # create matrix linear indices, corresponding x and y values
            c_cidx = c[cidx_idx[0], cidx_idx[1]]
            r_cidx = r[cidx_idx[0], cidx_idx[1]]
            cidx = np.array([cidx_row, cidx_col])

            cidx_flat = cidx_row * C.shape[1] + cidx_col

            # create matrix indices
            vals = np.vstack((cidx_idx[0], cidx_idx[1], c_cidx - cent_x, r_cidx - cent_y))

            # convert to polar and sort based on theta
            [_, theta] = coor.cart2pol(vals[2], vals[3])
            vals = np.vstack((vals, theta))

            order = vals[4].argsort()
            sortedvals = vals[:, order]

            circ_vec = char_binary[sortedvals[0].astype(int), sortedvals[1].astype(int)]

            # cv2.imshow('img'+str(i), circ_vec)
            # cv2.waitKey(0)

            # morphology filling
            kernel = np.ones((2, 1), np.uint8)
            circ_vec = cv2.morphologyEx(circ_vec, cv2.MORPH_OPEN, kernel)

            # kernel = np.array([[1, 0], [1, 0]]).astype(np.uint8)
            # circ_vec = np.array(skimorph.opening(np.array(circ_vec, dtype=np.uint8).reshape(circ_vec.shape[0],1), kernel))

            cntVec = np.vstack((1, 1, circ_vec)).astype(int)
            cntVec = np.reshape(cntVec, (cntVec.shape[0],))
            cntVec = cntVec.tolist()
            strcntVec = ''.join(str(e) for e in cntVec)
            strPattern = '00'
            cnt = [i for i in range(len(strcntVec)) if strcntVec.startswith(strPattern, i)]
            if cnt == []:
                count = 0
            else:
                a = np.diff([1] + cnt)
                idx_count = []
                for i_a in a:
                    if i_a == 1:
                        idx_count += [False]
                    else:
                        idx_count += [True]

                cnt = np.array(cnt)
                count = len(cnt[idx_count])
            coding_count[i] = count

            # Save count to identifier vector
            char_id[:, i] = coding_count[i]

            # cv2.imshow('img'+str(i), circ_vec)
            # cv2.waitKey(0)

            # Find 2 longest arcs of background and divide diff by total length
            # If 1s (background) wrap around, move all 1s to one side
            circ = len(circ_vec)
            if circ_vec[0] == 1 and circ_vec[-1] == 1:
                idx = np.argwhere(np.array(circ_vec) == 0)
                if idx.size == 0:
                    circ_vec = np.array([])
                    # Vector of length of each section of 1s
                    # Pad with 0s for diff
                    B = np.array([0, 0])
                else:
                    idx = idx[:, 0][-1]
                    circ_vec = np.vstack((circ_vec[idx::], circ_vec[0:idx]))
                    # circ_vec = [circ_vec[(idx + 1)::]] + circ_vec[0:idx]
                    # Vector of length of each section of 1s
                    # Pad with 0s for diff
                    B = np.vstack((0, np.array(circ_vec), 0))
            else:
                B = np.vstack((0, np.array(circ_vec), 0))
            B = np.array(B).T
            bgrd_len = np.array(np.where(np.diff(B) == -1)) - np.array(np.where(np.diff(B) == 1))
            # bgrd_len = find(diff(B) == -1) - find(diff(B) == 1);

            # If no crossing, set ratio to 0
            if coding_count[i] < 1:
                d2 = 0
                d1 = 0
            else:
                # Get 2 longest sections (arcs)
                d2 = bgrd_len[1, :].max()
                d2_get_max = np.array(np.where(bgrd_len == d2))
                # idx_d2 = int(d2_get_max[1])
                # [d2, idx_d2] = np.max(bgrd_len[1, :])
                # bgrd_len[1, idx_d2] = -1
                bgrd_len[d2_get_max[0], d2_get_max[1]] = -1
                d1 = np.max(bgrd_len[1, :])

            # Find ratio of difference of 2 longest arcs by circumference
            coding_ratio[i] = (d2 - d1) / circ
            if i > 1:   # Only keep k-1 largest ratios
                if ~np.isnan(coding_ratio[i]):
                    char_id[:, k+i-1] = coding_ratio[i]

        else:
            c = 0

    # Hu invariant moment
    char_hu_moment = cv2.HuMoments(char_moment)
    char_id[:, 2*k::] = char_hu_moment[1::].T

    # print(char_id)
    # a = hu_moments.cal_hm(char_img_inv)
    # print(a)
    # print(char_hu_moment)
    sio.savemat('./char_identifier.mat', {'identifier': char_id})


img = cv2.imread('./Segment/seg1.jpg', -1)
create_char_id(img)
