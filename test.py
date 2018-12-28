import scipy.io as sio
from src.match_character import *
from src.assemble_eq import *
from src.squeeze_matlab_matrix import *

[chars] = sio.loadmat('data/red_charPalette_withText_demo2.mat')['chars']
X_orig = sio.loadmat('data/red_charPalette_Classifier_demo2.mat')['X_orig']
[eq_chars] = sio.loadmat('data/eq_chars.mat')['eq_chars']

chars = squeeze_matlab_matrix(chars)
eq_chars = squeeze_matlab_matrix(eq_chars)
X_orig = squeeze_matlab_matrix(X_orig)

eq_chars = fn_match_character(eq_chars, X_orig, chars)

eq_string = fn_assemble_eq(eq_chars)

print(eq_string)