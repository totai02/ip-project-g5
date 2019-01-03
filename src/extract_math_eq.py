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


def extract_mat_eq(image, show_fig=False):
    if image is None:
        return "Image error!"

    # Optimize page and binarize
    eq_bin = fn_lighting_compensation(image)

    # Deskew Equation
    eq_deskew = fn_deskew(eq_bin, True)

    # Segment Equation Characters and Create Identifier
    eq_chars = fn_segment(eq_deskew)

    # Create Identifier
    for i in range(len(eq_chars)):
        eq_chars[i]['ident'] = fn_create_ident(eq_chars[i]['img'])

    # Match Characters(pass in struct of segmented chars and related info
    eq_chars = fn_match_character(eq_chars, X_orig, chars)

    # Assembled Equation
    eq_string = fn_assemble_eq(eq_chars)

    # for i in range(len(eq_chars)):
    #     cv2.imshow(str(i), eq_chars[i]['img'].astype(float))
    #
    # cv2.waitKey()
    #
    # print(fn_create_ident(eq_chars[1]['img']))
    # cv2.waitKey()

    return eq_string






