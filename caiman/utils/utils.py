""" pure utilitaries(other)

 all of other usefull functions

See Also
------------
https://docs.python.org/2/library/urllib.html

"""
#\package Caiman/utils
#\version   1.0
#\bug
#\warning
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2015
#\author: andrea giovannucci
#\namespace utils
#\pre none
from __future__ import print_function

import numpy as np
from scipy.ndimage.filters import gaussian_filter


#%% Generate data
def gen_data(dims=(48, 48), N=10, sig=(3, 3), tau=1., noise=.3, T=2000,
             framerate=30, firerate=.5, seed=3, cmap=False, truncate=np.exp(-2),
             difference_of_Gaussians=True, fluctuating_bkgrd=[50, 300]):
    bkgrd = 10  # fluorescence baseline
    np.random.seed(seed)
    boundary = 4
    M = int(N * 1.5)
    # centers = boundary + (np.array(GeneralizedHalton(2, seed).get(M)) *
    #                       (np.array(dims) - 2 * boundary)).astype('uint16')
    centers = boundary + (np.random.rand(M, 2) *
                          (np.array(dims) - 2 * boundary)).astype('uint16')
    trueA = np.zeros(dims + (M,), dtype='float32')
    for i in range(M):
        trueA[tuple(centers[i]) + (i,)] = 1.
    if difference_of_Gaussians:
        q = .75
        for n in range(M):
            s = (.67 + .33 * np.random.rand(2)) * np.array(sig)
            tmp = gaussian_filter(trueA[:, :, n], s)
            trueA[:, :, n] = np.maximum(tmp - gaussian_filter(trueA[:, :, n], q * s) *
                                        q**2 * (.2 + .6 * np.random.rand()), 0)

    else:
        for n in range(M):
            s = [ss * (.75 + .25 * np.random.rand()) for ss in sig]
            trueA[:, :, n] = gaussian_filter(trueA[:, :, n], s)
    trueA = trueA.reshape((-1, M), order='F')
    trueA *= (trueA >= trueA.max(0) * truncate)
    trueA /= np.linalg.norm(trueA, 2, 0)
    keep = np.ones(M, dtype=bool)
    overlap = trueA.T.dot(trueA) - np.eye(M)
    while(keep.sum() > N):
        keep[np.argmax(overlap * np.outer(keep, keep)) % M] = False
    trueA = trueA[:, keep]
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueS[:, 0] = 0
    for i in range(N // 2):
        trueS[i, :500 + i * T // N * 2 // 3] = 0
    trueC = trueS.astype('float32')
    for i in range(N):
        gamma = np.exp(-1. / (tau * framerate))  # * (.9 + .2 * np.random.rand())))
        for t in range(1, T):
            trueC[i, t] += gamma * trueC[i, t - 1]

    if fluctuating_bkgrd:
        K = np.array([[np.exp(-(i - j)**2 / 2. / fluctuating_bkgrd[0]**2)
                       for i in range(T)] for j in range(T)])
        ch = np.linalg.cholesky(K + 1e-10 * np.eye(T))
        truef = 1e-2 * ch.dot(np.random.randn(T)).astype('float32') / bkgrd
        truef -= truef.mean()
        truef += 1
        K = np.array([[np.exp(-(i - j)**2 / 2. / fluctuating_bkgrd[1]**2)
                       for i in range(dims[0])] for j in range(dims[0])])
        ch = np.linalg.cholesky(K + 1e-10 * np.eye(dims[0]))
        trueb = 3 * 1e-2 * \
            np.outer(*ch.dot(np.random.randn(dims[0], 2)).T).ravel().astype('float32')
        trueb -= trueb.mean()
        trueb += 1
    else:
        truef = np.ones(T, dtype='float32')
        trueb = np.ones(np.prod(dims), dtype='float32')
    trueb *= bkgrd
    Yr = np.outer(trueb, truef) + noise * np.random.randn(
        * (np.prod(dims), T)).astype('float32') + trueA.dot(trueC)

    if cmap:
        import matplotlib.pyplot as plt
        import caiman as cm
        Y = np.reshape(Yr, dims + (T,), order='F')
        Cn = cm.local_correlations(Y)
        plt.figure(figsize=(20, 3))
        plt.plot(trueC.T)
        plt.figure(figsize=(20, 3))
        plt.plot((trueA.T.dot(Yr - bkgrd) / np.sum(trueA**2, 0).reshape(-1, 1)).T)
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.scatter(*centers[keep].T[::-1], c='g')
        plt.scatter(*centers[~keep].T[::-1], c='r')
        plt.imshow(Y[:T // 10 * 10].reshape(dims + (T // 10, 10)).mean(-1).max(-1), cmap=cmap)
        plt.title('Max')
        plt.subplot(132)
        plt.scatter(*centers[keep].T[::-1], c='g')
        plt.scatter(*centers[~keep].T[::-1], c='r')
        plt.imshow(Y.mean(-1), cmap=cmap)
        plt.title('Mean')
        plt.subplot(133)
        plt.scatter(*centers[keep].T[::-1], c='g')
        plt.scatter(*centers[~keep].T[::-1], c='r')
        plt.imshow(Cn, cmap=cmap)
        plt.title('Correlation')
        plt.show()
    return Yr, trueC, trueS, trueA, trueb, truef, centers, dims
