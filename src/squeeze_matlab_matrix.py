
def squeeze_matlab_matrix(mat):
    mat = mat.squeeze()
    for r in range(len(mat)):
        for c in range(len(mat[r])):
            mat[r][c] = mat[r][c].squeeze()

    return mat
