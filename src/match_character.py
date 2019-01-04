from sklearn.neighbors import NearestNeighbors


def fn_match_character(eq_chars, X_orig, chars, n_nb = 1):
    knn_search = NearestNeighbors(n_neighbors=n_nb, metric='cityblock')
    knn_search.fit(X_orig[:, :-1])

    char_id = []
    for i in range(len(eq_chars)):
        idx_matched = knn_search.kneighbors([eq_chars[i]['ident']], return_distance=False)
        eq_chars[i]['char'] = chars[0, int(X_orig[idx_matched, -1])]['char'].squeeze()
        char_id.append(int(X_orig[idx_matched, -1]))

    return eq_chars, char_id
