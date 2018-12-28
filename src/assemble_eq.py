import numpy as np


def fn_assemble_eq(chars):
    control = help_create_control()

    limit_control = np.array(('int', 'sum', 'prod', 'lim'))
    space = 10
    eq_string = ''
    num_chars = len(chars)

    boxes = np.zeros([4, num_chars])
    for i in range(num_chars):
        boxes[:, i] = chars[i]['boundingbox']

    idxs = np.argsort(boxes[0, :])
    chars = chars[idxs]
    boxes = boxes[:, idxs]
    # Do preprocessing on chars

    # error

    # variables for super/subscript checks
    prev_centroid_y_coord = 0
    prev_fraction = False
    is_super = False
    is_sub = False

    # Loop through chars and create string
    i = 0

    while i < num_chars:
        detected = str(chars[i]['char'])

        ul_x_coord = boxes[0, i]
        ur_x_coord = ul_x_coord + boxes[2, i]

        overlap_idx = (boxes[0, :] >= ul_x_coord) & (boxes[0, :] <= ur_x_coord)
        cur_overlap_boxes = boxes[:, overlap_idx]
        cur_overlap_top = np.min(cur_overlap_boxes[1, :])
        cur_overlap_bottom = np.max(cur_overlap_boxes[1, :] + cur_overlap_boxes[3, :])
        total_height = cur_overlap_bottom - cur_overlap_top

        overlap_idx = (boxes[0, :] >= ul_x_coord) & (boxes[0, :] <= ur_x_coord) & (boxes[3, :] < 0.7 * total_height)

        cur_overlap_boxes = boxes[:, overlap_idx]
        ul_list = cur_overlap_boxes[0, :]
        ur_list = cur_overlap_boxes[0, :] + cur_overlap_boxes[2, :]
        top_list = cur_overlap_boxes[1, :]
        bottom_list = cur_overlap_boxes[1, :] + cur_overlap_boxes[3, :]
        while ul_list.size != 0 and (np.min(ul_list) < ul_x_coord or np.max(ur_list) > ur_x_coord):
            ul_x_coord = np.min(ul_list)
            ur_x_coord = np.max(ur_list)
            total_height = np.max(bottom_list) - np.min(top_list)
            overlap_idx = (boxes[0, :] >= ul_x_coord) & (boxes[0, :] <= ur_x_coord) & (boxes[3, :] < 0.7 * total_height)
            cur_overlap_boxes = boxes[:, overlap_idx]
            ul_list = cur_overlap_boxes[0, :]
            ur_list = cur_overlap_boxes[0, :] + cur_overlap_boxes[2, :]

        overlap_idx[i] = False
        num_overlaps = np.sum(overlap_idx)

        if detected == 'sqrt':
            sqrt_chars = chars[overlap_idx]
            sqrt_str = fn_assemble_eq(sqrt_chars)
            eq_string += '\sqrt{' + sqrt_str + '}'
            i = i + num_overlaps + 1
            continue

        def help_get_char(_detected, index, _chars, _dist_to_next):

            if np.any(control == _detected) or np.any(limit_control == _detected):
                _detected = '\\' + _detected + ' '

            if dist_to_next and help_is_letter(_detected) and help_is_letter(chars[index + 1]['char']):
                if dist_to_next >= space:
                    _detected += '\\,'

            return _detected

        if num_overlaps == 0:
            if (i > 0) and not prev_fraction:
                ll_corner = boxes[1, i] + boxes[3, i]
                ul_corner = boxes[1, i]
                if ll_corner <= np.ceil(prev_centroid_y_coord):
                    if detected != '-' or ll_corner < (prev_centroid_y_coord - 5 * boxes[3, i]):
                        is_super = True
                        eq_string += '^{'

                        prev_centroid_y_coord = chars[i]['centroid'][1]

                        if i + 1 < num_chars:
                            dist_to_next = boxes[0, i + 1] - ur_x_coord
                        else:
                            dist_to_next = None

                        detected = help_get_char(detected, i, chars, dist_to_next)

                elif ul_corner >= np.floor(prev_centroid_y_coord):
                    if is_super:
                        is_super = False
                        eq_string = eq_string.strip() + '}'
                        prev_fraction = False

                        prev_centroid_y_coord = chars[i]['centroid'][1]

                        if i + 1 < num_chars:
                            dist_to_next = boxes[0, i + 1] - ur_x_coord
                        else:
                            dist_to_next = None

                        detected = help_get_char(detected, i, chars, dist_to_next)
                    else:
                        is_sub = True
                        eq_string = eq_string.strip() + '_{'

                        if i + 1 < num_chars:
                            dist_to_next = boxes[0, i + 1] - ur_x_coord
                        else:
                            dist_to_next = None

                        detected = help_get_char(detected, i, chars, dist_to_next)

                        prev_centroid_y_coord = chars[i]['centroid'][1]
                else:
                    prev_fraction = False

                    prev_centroid_y_coord = chars[i]['centroid'][1]

                    if i + 1 < num_chars:
                        dist_to_next = boxes[0, i + 1] - ur_x_coord
                    else:
                        dist_to_next = None

                    detected = help_get_char(detected, i, chars, dist_to_next)
            else:

                prev_fraction = False

                prev_centroid_y_coord = chars[i]['centroid'][1]

                if i + 1 < num_chars:
                    dist_to_next = boxes[0, i + 1] - ur_x_coord
                else:
                    dist_to_next = None

                detected = help_get_char(detected, i, chars, dist_to_next)

            eq_string += detected

        elif num_overlaps == 0:
            assert overlap_idx[i + 1]

            overlap_ul = boxes[0, i + 1]

            if abs(overlap_ul - ur_x_coord) <= 1:
                if np.any(control == detected):
                    detected = '\\' + detected + ' '

                if i + 1 < num_chars and help_is_letter(detected) and help_is_letter(chars[i + 1]['char']):
                    dist_to_next = boxes[0, i + 1] - ur_x_coord
                    if dist_to_next >= space:
                        detected += '\\,'

                eq_string += detected

                prev_centroid_y_coord = chars[i]['centroid'][1]

            else:
                overlap_char = chars[i + 1]['char']
                if detected == '-' and overlap_char == '-':
                    eq_string += '='

                    prev_centroid_y_coord = 1/2 * (chars[i]['centroid'][1] + chars[i + 1]['centroid'][1])
                i = i + 1
        else:
            overlap_idx[i] = True
            overlap_chars = chars[overlap_idx]
            overlap_centroids = np.zeros([2, len(overlap_chars)])
            overlap_boxes = np.zeros([4, len(overlap_chars)])
            bar_idx = []
            bar_width = []
            limit_idx = []
            limit_height = []
            for overlap_i in range(len(overlap_chars)):
                overlap_centroids[:, overlap_i] = overlap_chars[overlap_i]['centroid']
                overlap_boxes[:, overlap_i] = overlap_chars[overlap_i]['boundingbox']
                if overlap_chars[overlap_i]['char'] == '-':
                    bar_idx.append(overlap_i)
                    bar_width.append(overlap_chars[overlap_i]['boundingbox'][2])
                elif np.any(limit_control == overlap_chars[overlap_i]['char']):
                    limit_idx.append(overlap_i)
                    limit_height.append(overlap_chars[overlap_i]['boundingbox'][3])

            ul_list = overlap_boxes[0, :]
            ur_list = overlap_boxes[0, :] + overlap_boxes[2, :]
            total_width = np.max(ur_list) - np.min(ul_list)

            bar_idx = np.array(bar_idx)[np.array(bar_width) >= 0.8 * total_width]

            if len(bar_idx) == 1:
                prev_fraction = True
                frac_bar = overlap_chars[bar_idx[0]]
                frac_y_coord = frac_bar['centroid'][1]
                frac_bar_height = frac_bar['boundingbox'][3]

                if prev_centroid_y_coord and frac_y_coord < prev_centroid_y_coord - 7 * frac_bar_height:
                    is_super = True
                    eq_string = eq_string.strip() + '^{'

                prev_centroid_y_coord = frac_y_coord

                numer_idx = overlap_centroids < frac_y_coord
                numer_idx = numer_idx[1, :]
                denom_idx = overlap_centroids > frac_y_coord
                denom_idx = denom_idx[1, :]

                numer_eq = overlap_chars[numer_idx]
                denom_eq = overlap_chars[denom_idx]

                numer_str = fn_assemble_eq(numer_eq)
                denom_str = fn_assemble_eq(denom_eq)

                detected = '\\frac{' + numer_str + '}{' + denom_str + '}'

                if is_super:
                    is_super = False
                    detected += '}'

                i = overlap_idx.nonzero()[0][-1]

            elif len(limit_idx) != 0:
                dom_idx = np.argmax(limit_height)
                dom_idx = np.array(limit_idx)[dom_idx]

                limit_char = overlap_chars[dom_idx]
                limit_y_coord = limit_char['centroid'][1]

                prev_centroid_y_coord = limit_y_coord

                top_idx = overlap_centroids < limit_y_coord
                top_idx = top_idx[1, :]
                bottom_idx = overlap_centroids > limit_y_coord
                bottom_idx = bottom_idx[1, :]

                detected = '\\' + str(limit_char['char']) + '\\limits'

                if np.any(bottom_idx):
                    bottom_eq = overlap_chars[bottom_idx]
                    bottom_str = fn_assemble_eq(bottom_eq)
                    detected += '_{' + bottom_str + '}'

                if np.any(top_idx):
                    top_eq = overlap_chars[top_idx]
                    top_str = fn_assemble_eq(top_eq)
                    detected += '^{' + top_str + '}'

                i = overlap_idx.nonzero()[0][-1]
            eq_string += detected
        i = i + 1

    if is_super or is_sub:
        eq_string = eq_string.strip() + '}'

    return eq_string


def help_is_letter(detected):
    detected = str(detected)
    return len(detected) == 1 and str(detected).isalpha()


def help_create_control():
    return np.array((
        'alpha',
        'beta',
        'gamma',
        'delta',
        'epsilon',
        'zeta',
        'eta',
        'theta',
        'iota',
        'kappa',
        'lambda',
        'mu',
        'nu',
        'xi',
        'pi',
        'rho',
        'sigma',
        'tau',
        'upsilon',
        'phi',
        'chi',
        'psi',
        'omega',
        'Alpha',
        'Beta',
        'Gamma',
        'Delta',
        'Epsilon',
        'Zeta',
        'Eta',
        'Theta',
        'Iota',
        'Kappa',
        'Lambda',
        'Mu',
        'Nu',
        'Xi',
        'Pi',
        'Rho',
        'Sigma',
        'Tau',
        'Upsilon',
        'Phi',
        'Psi',
        'Omega',
        'int',
        'rightarrow',
        'infty',
    ))


# def help_preprocess_chars(chars, boxes):
#     i = 0
#     new_chars = np.zeros_like(chars)
#     new_chars[0] = chars[0]
#     while i < len(chars):
#         detected = chars[i]['char']
#
#         if detected == 'l':
#             start_x_coord = chars[i]['boundingbox'][0][0]
#             end_x_coord = chars[i]['boundingbox'][0][2] * 3 + start_x_coord
#             following_idx = np.argwhere((boxes[0, :] > start_x_coord) & (boxes[0, :] < end_x_coord))
#             following = chars[following_idx]
#             lim_chars_idx = (following['char'] == 'l') | (following['char'] == '.') | (following['char'] == 'm')
#             lim_chars_idx = following_idx[lim_chars_idx]
