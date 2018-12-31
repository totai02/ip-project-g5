import scipy.io as sio
from src.match_character import *
from src.assemble_eq import *
from src.squeeze_matlab_matrix import *
from src.segment import *
from src.lighting_compensation import *
from src.create_char_ident import *
import cv2

img = cv2.imread('data/Equations/Clean/eq1_hr.jpg')
bw_img = fn_lighting_compensation(img)
eq_chars = fn_segment(bw_img)

for i in range(len(eq_chars)):
    eq_chars[i]['ident'] = create_char_id(eq_chars[i]['img'].astype(np.uint8) * 255)