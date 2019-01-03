import numpy as np
from skimage import morphology, measure
from scipy.spatial import ConvexHull

Character = np.dtype([('centroid', 'O'), ('boundingbox', 'O'), ('img', 'O'), ('ident', 'O'), ('char', 'O')])


def fn_segment(eq):
    eq_inv = np.ones_like(eq) - eq
    se = np.ones((3, 3))
    exp = morphology.binary_erosion(eq_inv, se)
    eq_edges = np.logical_xor(exp, eq_inv)

    label_eq = measure.label(eq_edges, connectivity=eq_edges.ndim)
    props = measure.regionprops(label_eq)

    loc = np.array([np.array(item.centroid) for item in props])
    loc = loc[:, [1, 0]]

    coords = [np.array(item.coords) for item in props]
    hulls = [ConvexHull(coord) for coord in coords]
    ch = []
    for i in range(len(hulls)):
        ch.append(coords[i][hulls[i].vertices, :])
    ch = np.array(ch)

    boundingboxes = np.array([np.array(item.bbox) for item in props])
    boundingboxes = boundingboxes[:, [1, 0, 3, 2]]
    boundingboxes[:, 2] = boundingboxes[:, 2] - boundingboxes[:, 0]
    boundingboxes[:, 3] = boundingboxes[:, 3] - boundingboxes[:, 1]

    imgs = np.array([np.array(item.image) for item in props])

    idx = []
    for i in range(len(ch)):
        for j in range(len(ch)):
            if np.sum(~measure.points_in_poly(ch[i], ch[j])) == 0 and i != j:
                bb_outer = boundingboxes[j, :]
                bb_inner = boundingboxes[i, :]

                bb_inner[0] = bb_inner[0] - bb_outer[0]
                bb_inner[1] = bb_inner[1] - bb_outer[1]

                outer = imgs[j]

                up = np.sum(outer[0:bb_inner[1] + bb_inner[3], bb_inner[0]:bb_inner[0] + bb_inner[2]], 0)
                down = np.sum(outer[bb_inner[1]:, bb_inner[0]:bb_inner[0] + bb_inner[2]], 0)
                left = np.sum(outer[bb_inner[1]:bb_inner[1] + bb_inner[3], 0:bb_inner[0] + bb_inner[2]], 1)
                right = np.sum(outer[bb_inner[1]:bb_inner[1] + bb_inner[3], bb_inner[0]:], 1)
                if sum(up == 0) + sum(down == 0) + sum(left == 0) + sum(right == 0) == 0:
                    idx.append(i)

    idx = np.unique(np.array(idx))

    if len(idx) != 0:
        for i in reversed(range(len(idx))):
            loc = np.delete(loc, idx[i], 0)
            ch = np.delete(ch, idx[i], 0)
            boundingboxes = np.delete(boundingboxes, idx[i], 0)
            imgs = np.delete(imgs, idx[i], 0)

    characters = np.empty(len(loc), dtype=Character)

    for i in range(len(loc)):
        characters[i]['centroid'] = loc[i]
        characters[i]['boundingbox'] = boundingboxes[i]

        region = eq_inv[boundingboxes[i, 1]:boundingboxes[i, 1] + boundingboxes[i, 3],
                 boundingboxes[i, 0]:boundingboxes[i, 0] + boundingboxes[i, 2]]

        label_objects = measure.label(region, connectivity=region.ndim)
        props = measure.regionprops(label_objects)
        if len(props) > 1:
            num_pixels = [obj.area for obj in props]
            idx = np.argmax(num_pixels)
            region.fill(True)
            region[props[idx].coords[:, 0], props[idx].coords[:, 1]] = False

            th_ratio = 1
            th_sol = 0.2
            sqrt_ratio = 0.7812
            ratio = boundingboxes[i, 2] / boundingboxes[i, 3]

            if ratio > th_ratio and props[idx].solidity < th_sol:
                new_w = int(round(boundingboxes[i, 3] * sqrt_ratio))
                region = region[:, 0:new_w]
                idx = np.argwhere(region[:, -1] == 0).squeeze()
                last = idx[-1]
                region = region[last + 1:, :]

            characters[i]['img'] = region
        else:
            characters[i]['img'] = eq[boundingboxes[i, 1]:boundingboxes[i, 1] + boundingboxes[i, 3],
                                   boundingboxes[i, 0]:boundingboxes[i, 0] + boundingboxes[i, 2]]

        characters[i]['ident'] = None
        characters[i]['char'] = None

    return characters
