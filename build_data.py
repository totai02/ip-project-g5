import scipy.io as sio
from src.create_ident import *

Character = np.dtype([('centroid', 'O'), ('boundingbox', 'O'), ('img', 'O'), ('ident', 'O'), ('char', 'O')])


new_chars = np.empty(116, dtype=Character)
X_orig = np.zeros((116, 23))

chars = sio.loadmat("data/chars")['chars']

for i in range(len(chars)):
    print("Character " + str(i + 1) + ":")
    new_chars[i]['centroid'] = chars[i]['centroid']
    new_chars[i]['boundingbox'] = chars[i]['boundingbox']
    new_chars[i]['img'] = chars[i]['img']
    new_chars[i]['ident'] = np.empty([])
    X_orig[i, :-1] = fn_create_ident(new_chars[i]['img'])
    X_orig[i, -1] = i
    new_chars[i]['char'] = chars[i]['char']

sio.savemat("chars.mat", {"chars": new_chars, "X_orig": X_orig})

