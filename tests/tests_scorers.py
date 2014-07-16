import numpy as np

def test_specificity():
    specificity(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]))
    specificity(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0]))
    specificity(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 1]))
    specificity(np.array([1, 1, 0, 0]), np.array([.7, .7, .2, .2]))