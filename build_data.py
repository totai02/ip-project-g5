import scipy.io as sio
from src.create_ident import *

Character = np.dtype([('centroid', 'O'), ('boundingbox', 'O'), ('img', 'O'), ('ident', 'O'), ('char', 'O')])


new_chars = np.empty(116, dtype=Character)
X_orig = np.zeros((116, 23))

chars = sio.loadmat("data/chars")['chars']

for i in range(len(chars[0])):
    print("Character " + str(i + 1) + ":")
    new_chars[i]['centroid'] = chars[0, i]['centroid']
    new_chars[i]['boundingbox'] = chars[0, i]['boundingbox']
    new_chars[i]['img'] = chars[0, i]['img']
    new_chars[i]['ident'] = np.empty([])
    X_orig[i, :-1] = fn_create_ident(new_chars[i]['img'])
    X_orig[i, -1] = i
    new_chars[i]['char'] = chars[0, i]['char']

sio.savemat("chars.mat", {"chars": new_chars, "X_orig": X_orig})

