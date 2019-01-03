from scipy import misc
import os
import cv2

path = "data/Equations/Clean/"
acc_path = "data/accuracy/"

if not os.path.exists(acc_path):
    os.makedirs(acc_path)

for i in range(1, 14):
    filename = path + "eq{}_hr.jpg".format(i)
    img = cv2.imread(filename)

    new_dir = acc_path + "origin/"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    cv2.imwrite(acc_path + "origin/" + "eq{}_hr.jpg".format(i), img)
    img_inv = 255 - img

    for angle in [-15, -10, -5, 5, 10, 15]:
        new_dir = acc_path + "rotate{}".format(angle)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        new_img = misc.imrotate(img_inv, angle)
        new_img = 255 - new_img
        cv2.imwrite(acc_path + "rotate{}/eq{}_hr.jpg".format(angle, i), new_img)

    for scale in [0.5, 2.0]:
        new_dir = acc_path + "scale{}".format(scale)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        width = round(img.shape[0] * scale)
        height = round(img.shape[1] * scale)
        new_img = misc.imresize(img, (width, height))
        cv2.imwrite(acc_path + "scale{}/eq{}_hr.jpg".format(scale, i), new_img)
