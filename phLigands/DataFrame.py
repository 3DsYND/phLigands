import numpy as np

def read_file(path, delimiter="\t", names=None):
    return np.transpose(np.genfromtxt(path, delimiter=delimiter, names=names))
