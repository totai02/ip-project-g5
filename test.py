from src.extract_math_eq import *
import cv2

img = cv2.imread("data/Equations/Clean/eq4_hr.jpg")
eq_string, _ = extract_mat_eq(img, False)
print(eq_string)
