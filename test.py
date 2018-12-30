import scipy.io as sio
from src.match_character import *
from src.assemble_eq import *
from src.squeeze_matlab_matrix import *
from src.lighting_compensation import *
import cv2

img = cv2.imread('data/Equations/Images/eq2_hr.jpg')
bw_img = fn_lighting_compensation(img)

cv2.imshow('test', bw_img.astype(np.uint8) * 255)
cv2.waitKey()

