import numpy as np
import pickle

centroid = []
boundingbox = []
img = []
ident = []
character = []

char_palette = {}

with open('data/centroid.txt') as f:
    for line in f:
        arr = line.split(',')
        centroid.append(np.array([float(num) for num in arr]))

with open('data/boundingbox.txt') as f:
    for line in f:
        arr = line.split(',')
        boundingbox.append(np.array([int(num) for num in arr]))

with open('data/img.txt') as f:
    for line in f:
        arr = line.split(',')
        img.append(np.array([int(num) for num in arr]))

with open('data/ident.txt') as f:
    for line in f:
        arr = line.split(',')
        ident.append(np.array([float(num) for num in arr]))

with open('data/character.txt') as f:
    for line in f:
        character.append(line)

char_palette['centroid'] = centroid
char_palette['boundingbox'] = boundingbox
char_palette['ident'] = ident
char_palette['character'] = character

with open('char_palette_withText.pickle', 'wb') as f:
    pickle.dump(char_palette, f, pickle.HIGHEST_PROTOCOL)

