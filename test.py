from src.extract_math_eq import *
import cv2

img = cv2.imread("data/Equations/Images/eq2_hr.jpg")
eq_string = extract_mat_eq(img, False)
print(eq_string)
