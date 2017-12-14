from os import path
import numpy as np
from .matlab import loadmat


def sbxinfo(filename):
    """Loads the .mat file associated with .sbx files to get formatting info, adding some more info as well"""
    info = loadmat(filename)['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2;
        info['factor'] = 1
    elif info['channels'] == 2:
        info['nChan'] = 1;
        info['factor'] = 2
    elif info['channels'] == 3:
        info['nChan'] = 1;
        info['factor'] = 2

    return info


def sbxread(filename, k=0, n_frames=np.inf):
    """Returns data array from .sbx files."""
    filename = path.splitext(filename)[0]  # strip out file extension.
    info = sbxinfo(filename + '.mat')

    # Determine number of frames in whole file
    max_idx = path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * info['factor'] / 4 - 1
    N = np.minimum(max_idx, n_frames)  # todo
    n_samples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Load Data from File
    with open(filename + '.sbx') as fo:
        fo.seek(k * n_samples, 0)
        x = np.fromfile(fo, dtype='uint16', count=int(n_samples / 2 * N))

    # Transform values and reshape array, then return it
    x = np.iinfo(np.uint16).max - x  # SBX files require subtracting the data from the max int16 to get the correct values.
    x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N)), order='F')[0]
    x = x.transpose([2, 1, 0])
    return x


def sbxreadskip(filename, skip):
    """
    Input:
     -----
    filename: str
         filename should be full path excluding .sbx
    """
    # Check if contains .sbx and if so just truncate
    filename = path.splitext(filename)[0]  # strip out file extension.
    info = sbxinfo(filename + '.mat')

    # Determine number of frames in whole file
    max_idx = path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * info['factor'] / 4 - 1
    N = max_idx + 1  # Last frame
    n_samples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Load Data from File
    with open(filename + '.sbx') as fo:
        for k in range(0, N, skip):
            fo.seek(k * n_samples, 0)
            tmp = np.iinfo(np.uint16).max - np.fromfile(fo, dtype='uint16', count=int(n_samples / 2 * 1))  # SBX files require subtracting the data from the max int16 to get the correct values.
            tmp = tmp.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(1)), order='F')
            x = tmp if k is 0 else np.concatenate((x, tmp), axis=3)[0]
    x = x.transpose([2, 1, 0])
    return x


def sbxshape(filename):
    """Returns 3-element shape tuple from an spx's associated .mat file."""
    info = sbxinfo(filename)
    max_idx = path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * info['factor'] / 4 - 1
    N = max_idx + 1  # Last frame
    x = (int(info['sz'][1]), int(info['recordsPerBuffer']), int(N))
    return x

