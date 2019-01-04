from src.extract_math_eq import *
import cv2

img = cv2.imread("data/Equations/Images/eq4_hr.jpg")
eq_string, _ = extract_mat_eq(img, True)
print(eq_string)
