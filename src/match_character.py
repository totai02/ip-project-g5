from sklearn.neighbors import NearestNeighbors
import numpy as np


def fn_match_character(eq_chars, X_orig, chars, n_nb = 1):
    knn_search = NearestNeighbors(n_neighbors=n_nb, metric='cityblock')
    knn_search.fit(X_orig[:, :-1])

    for i in range(len(eq_chars)):
        idx_matched = knn_search.kneighbors(np.array([eq_chars[i]['ident']]), return_distance=False)
        eq_chars[i]['char'] = chars[int(X_orig[idx_matched, -1])]['char']

    return eq_chars
