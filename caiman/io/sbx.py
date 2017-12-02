from os import path
import numpy as np
from . import loadmat


def sbxread(filename, k=0, n_frames=np.inf):
    """
    Input:
    ------
    filename: str
        filename should be full path excluding .sbx
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2;
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1;
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1;
        factor = 2

    # Determine number of frames in whole file
    max_idx = path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1

    # Paramters
    N = max_idx + 1  # Last frame
    N = np.minimum(max_idx, n_frames)

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
    fo.seek(k * nSamples, 0)
    ii16 = np.iinfo(np.uint16)
    x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
    x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N)), order='F')
    x = x[0, :, :, :]

    return x.transpose([2, 1, 0])


def sbxreadskip(filename, skip):
    """
    Input:
     -----
    filename: str
         filename should be full path excluding .sbx
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2;
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1;
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1;
        factor = 2

    # Determine number of frames in whole file
    max_idx = path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1

    # Paramters
    N = max_idx + 1  # Last frame
    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
    for k in range(0, N, skip):
        fo.seek(k * nSamples, 0)
        ii16 = np.iinfo(np.uint16)
        tmp = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * 1))

        tmp = tmp.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(1)), order='F')
        if k is 0:
            x = tmp;
        else:
            x = np.concatenate((x, tmp), axis=3)

    x = x[0, :, :, :]

    return x.transpose([2, 1, 0])


def sbxshape(filename):
    """
    Input:
     -----
     filename should be full path excluding .sbx
    """

    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2;
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1;
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1;
        factor = 2

    # Determine number of frames in whole file
    max_idx = os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1
    N = max_idx + 1  # Last frame
    x = (int(info['sz'][1]), int(info['recordsPerBuffer']), int(N))
    return x

