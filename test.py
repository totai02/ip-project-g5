import scipy.io as sio
from src.match_character import *
from src.assemble_eq import *
from src.squeeze_matlab_matrix import *
from src.segment import *
from src.lighting_compensation import *
from src.create_ident import *
import cv2

img = cv2.imread('data/Equations/Clean/demo_equation_hr.jpg')
bw_img = fn_lighting_compensation(img)
eq_chars = fn_segment(bw_img)

fn_create_ident(eq_chars[10]['img'])

