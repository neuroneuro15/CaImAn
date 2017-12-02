import numpy as np


def com(A, d1, d2):
    """Calculation of the center of mass for spatial components

     Inputs:
     ------
     A:   np.ndarray
          matrix of spatial components (d x K)

     d1:  int
          number of pixels in x-direction

     d2:  int
          number of pixels in y-direction

     Output:
     -------
     cm:  np.ndarray
          center of mass for spatial components (K x 2)
    """
    nr = np.shape(A)[-1]
    Coor = dict()
    Coor['x'] = np.kron(np.ones((d2, 1)), np.expand_dims(list(range(d1)), axis=1))
    Coor['y'] = np.kron(np.expand_dims(list(range(d2)), axis=1), np.ones((d1, 1)))
    cm = np.zeros((nr, 2))        # vector for center of mass
    cm[:, 0] = np.dot(Coor['x'].T, A) / A.sum(axis=0)
    cm[:, 1] = np.dot(Coor['y'].T, A) / A.sum(axis=0)

    return cm