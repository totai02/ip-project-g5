from src.extract_math_eq import *
import cv2

img = cv2.imread("data/Equations/Clean/demo_equation_hr.jpg")
extract_mat_eq(img)
