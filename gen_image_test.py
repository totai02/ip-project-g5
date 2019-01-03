from scipy import misc
import cv2

path = "data/Equations/Clean/"
acc_path = "data/accuracy/"

for i in range(1, 14):
    filename = path + "eq{}_hr.jpg".format(i)
    img = cv2.imread(filename)

    img_inv = 255 - img

    for angle in [5, 10, 15]:
        new_img = misc.imrotate(img_inv, angle)
        new_img = 255 - new_img
        cv2.imwrite(acc_path + "eq{}_hr_r{}.jpg".format(i, angle), new_img)

    for scale in [0.5, 2.0]:
        width = img.shape[0] * scale
        height = img.shape[1] * scale
        new_img = misc.imresize(img, (width, height))
        cv2.imwrite(acc_path + "eq{}_hr_s{}.jpg".format(i, angle), new_img)
