"""Sigmoid function"""

import numpy as np


def sigmoid(matrix):
    """Applies sigmoid function to NumPy matrix"""
    matrix = np.clip(matrix, -500, 500)
    return 1 / (1 + np.exp(-matrix))
