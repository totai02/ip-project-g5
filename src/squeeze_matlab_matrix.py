
def squeeze_matlab_matrix(mat):
    for r in range(len(mat)):
        for c in range(len(mat[r])):
            mat[r][c] = mat[r][c].squeeze()

    return mat
