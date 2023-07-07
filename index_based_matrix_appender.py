import numpy as np


def index_based_matrix_appender(matrix1, matrix2):
    new_matrix = np.full((matrix1.shape[0], matrix1.shape[1] + 1), np.nan)
    new_matrix[:,:-1] = matrix1

    for i, arr in enumerate(matrix2):
        start_index = arr[0]
        if i == 0:
            start_index = 0
        value = arr[1]

        if i < len(matrix2) - 1:
            end_index = matrix2[i + 1][0]
        else:
            end_index = len(matrix1)

        new_matrix[start_index:end_index, -1] = value

    return new_matrix




#Test data:

#matrix3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16],[14, 14, 15, 16],[16, 14, 15, 16]])
#matrix4 = np.array([[1, 100], [2, 200], [3, 300]])

#print(index_based_matrix_appender(matrix3, matrix4))
