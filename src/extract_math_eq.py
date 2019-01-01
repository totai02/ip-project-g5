import scipy.io as sio
from src.lighting_compensation import *
from src.deskew import *
from src.segment import *
from src.create_ident import *
from src.match_character import *
from src.assemble_eq import *
import cv2

data = sio.loadmat("data/chars")
chars = data['chars']
X_orig = data["X_orig"]


def extract_mat_eq(image):
    eq_bin = fn_lighting_compensation(image)
    eq_deskew = deskew(eq_bin)

    cv2.imshow("img", eq_deskew)
    cv2.waitKey()
