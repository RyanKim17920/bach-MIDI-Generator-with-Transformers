import numpy as np


def index_based_matrix_appender(matrix1, matrix2):
    """
    This function appends a value to each row in matrix1 based on the index and value provided by matrix2.

    Parameters:
    matrix1 (numpy.ndarray): A 2D numpy array where values will be appended.
    matrix2 (numpy.ndarray): A 2D numpy array where each sub-array consists of an index and a value.

    Returns:
    new_matrix (numpy.ndarray): A 2D numpy array which is the result of appending values to matrix1 based on matrix2.
    """

    # Initialize a new list to hold the modified rows of matrix1
    new_matrix = []

    # Iterate over each sub-array in matrix2
    for i, arr in enumerate(matrix2):
        # The first element of the sub-array is the start_index
        start_index = arr[0]

        if i == 0:
            start_index = 0
        # The second element of the sub-array is the value to be appended
        value = arr[1]

        # Check if it's not the last index in matrix2
        if i < len(matrix2) - 1:
            # If it's not the last index, the end_index is the first element of the next sub-array
            end_index = matrix2[i + 1][0]
        else:
            # If it's the last index, the end_index is the total number of rows in matrix1
            end_index = len(matrix1)

        # Append the value to all rows in matrix1 from start_index to end_index
        new_rows = np.c_[matrix1[start_index:end_index], np.full((end_index - start_index, 1), value)].tolist()

        # Add the new rows to new_matrix
        new_matrix.extend(new_rows)

    # Convert new_matrix to a numpy array before returning
    return np.array(new_matrix)



#Test data:

#matrix3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16],[14, 14, 15, 16],[13, 14, 15, 16]])
#matrix4 = np.array([[1, 100], [2, 200], [3, 300]])

#print(index_based_matrix_appender(matrix3, matrix4))
