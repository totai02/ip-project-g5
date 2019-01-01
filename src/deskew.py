from skimage.filters import median
import numpy as np
import cv2


def rotate(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def clustering(array_theta):
    array_cluster = []
    array_label = []
    dist = 0.1
    label = 1

    for theta in array_theta:
        has_appended = False

        if not array_cluster:
            array_cluster.append([label, theta])
        else:
            for label_cluster, theta_cluster in array_cluster:
                if abs(theta - theta_cluster) < dist and has_appended == False:
                    has_appended = True
                    array_cluster.append([label_cluster, theta])

            if not has_appended:
                label += 1
                array_cluster.append([label, theta])

    for cluster in array_cluster:
        array_label.append(cluster[0])

    return array_label, array_cluster


def deskew(image, softening=False):
    image = image.astype(np.uint8) * 255
    gray = cv2.medianBlur(image, 5)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    array_theta = []
    for line in lines:
        for rho, theta in line:
            array_theta.append(theta)

    unique_theta = np.unique(array_theta)
    array_label, array_cluster = clustering(unique_theta)
    unique_label, indices = np.unique(array_label, return_counts=True)
    max_count_label = unique_label[np.argmax(indices)]
    new_unique_theta = []
    new_array_theta = []
    for label, theta in array_cluster:
        if label == max_count_label:
            new_unique_theta.append(theta)

    for theta in array_theta:
        if theta in new_unique_theta:
            new_array_theta.append(theta)

    array_theta = new_array_theta
    unique_theta, indices = np.unique(array_theta, return_counts=True)
    theta = unique_theta[np.argmax(indices)]
    deskewing_angle = np.rad2deg(theta)
    deskewing_angle = deskewing_angle - 90

    deskewing_image = rotate(image, deskewing_angle)

    if softening:
        deskewing_image = soften_edges(deskewing_image, 5)

    return deskewing_image


def soften_edges(image, se_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    softened = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return softened
