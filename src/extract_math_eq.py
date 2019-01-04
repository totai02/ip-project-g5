import scipy.io as sio
from src.lighting_compensation import *
from src.deskew import *
from src.segment import *
from src.create_ident import *
from src.match_character import *
from src.assemble_eq import *
import matplotlib.pyplot as plt

data = sio.loadmat("data/chars")
chars = data['chars']
X_orig = data["X_orig"]


def extract_mat_eq(image, show_fig=False):
    if image is None:
        return "Image error!", None

    if show_fig:
        plt.figure(num=1)
        plt.imshow(image)
        plt.show(block=False)
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)

    # Optimize page and binarize
    eq_bin = fn_lighting_compensation(image)
    if show_fig:
        plt.figure(num=2)
        plt.imshow(eq_bin)
        plt.axis("off")
        plt.show(block=False)
        plt.draw()
        plt.pause(0.001)

    # Deskew Equation
    eq_deskew = fn_deskew(eq_bin, True)
    if show_fig:
        plt.figure(num=3)
        plt.imshow(eq_deskew)
        plt.axis("off")
        plt.show(block=False)
        plt.draw()
        plt.pause(0.001)

    # Segment Equation Characters and Create Identifier
    eq_chars = fn_segment(eq_deskew)
    if show_fig:
        plt.figure(num=4)
        n = np.ceil(np.sqrt(len(eq_chars)))
        for i in range(len(eq_chars)):
            plt.subplot(n, n, i + 1)
            plt.imshow(eq_chars[i]['img'])
            plt.axis("off")

        plt.show(block=False)
        plt.draw()
        plt.pause(0.001)

    # Create Identifier
    for i in range(len(eq_chars)):
        eq_chars[i]['ident'] = fn_create_ident(eq_chars[i]['img'])

    # Match Characters(pass in struct of segmented chars and related info
    eq_chars, char_id = fn_match_character(eq_chars, X_orig, chars)

    if show_fig:
        plt.figure(num=5)
        n = len(eq_chars)
        for i in range(len(eq_chars)):
            plt.subplot(2, n, i + 1)
            plt.imshow(eq_chars[i]['img'])
            plt.title(str(i + 1))
            plt.axis("off")
        for i in range(len(eq_chars)):
            plt.subplot(2, n, len(eq_chars) + i + 1)
            plt.imshow((chars[0, char_id[i]]['img']).astype(np.uint8))
            plt.title(eq_chars[i]['char'])
            plt.axis("off")

        plt.show()

    # Assembled Equation
    eq_string = fn_assemble_eq(eq_chars)

    # for i in range(len(eq_chars)):
    #     cv2.imshow(str(i), eq_chars[i]['img'].astype(float))
    #
    # cv2.waitKey()
    #
    # print(fn_create_ident(eq_chars[1]['img']))
    # cv2.waitKey()

    return eq_string, eq_chars






