import numpy as np

def add_column_to_2d_array(array, number):
    # Create a column with the same number of rows as the original array
    column = np.full((array.shape[0], 1), number)
    # Add the column to the array
    return np.append(array, column, axis=1)