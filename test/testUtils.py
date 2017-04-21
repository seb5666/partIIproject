import numpy as np

#checks if n+1 dimensional numpy array a contains n-dimensional array b
def contains(a, b):
    for row in a:
        if np.array_equal(row, b):
            return True
    return False

#returns the index of n-dimensional array b in a. -1 if such does not exist
def index_of(a, b):
    for i in range(len(a)):
        row = a[i]
        if np.array_equal(row, b):
            return i
    return -1
