# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:33:35 2016

@author: agiovann
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from os import path


def gen_memmap_fname(base_filename, shape, order):
    fname, ext = path.splitext(base_filename)
    fname_tot = fname + '_' + order + '_' + '_'.join(map(str, shape))
    fname_tot = fname_tot + ext if ext else fname_tot + '.mmap_caiman'
    return fname_tot


def read_props_from_mmemap_fname(filename):
    extension = path.splitext(filename)[1]
    if not 'mmap_caiman' in extension:
        fdata = path.basename(filename).split('_')[1:]
        fname, order, shape = fdata[0], fdata[1], fdata[2:]
    else:
        d1, d2, d3, order, T = [int(part) for part in path.basename(filename).split('_')[-9::2]]
        # shape = (d1, d2) if d3 == 1 else (d1, d2, d3)
        shape = (d1 * d2 * d3, T)
    return shape, order


def load_memmap(filename, mode='r', in_memory=True):
    """Returns (Numpy.mmap object, shape tuple, frame count) from a file created by save_memmap()."""
    shape, order = read_props_from_mmemap_fname(filename)
    Yr = np.memmap(filename, mode=mode, shape=shape, dtype=np.float32, order=order)
    Yr = np.array(Yr) if in_memory else Yr
    return Yr, shape


def save_memmap(array, base_filename, order='F', n_chunks=1):
    """Saves efficiently an array into a Numpy memory mappable file.

    Parameters:
    ----------
        base_filename: string
            filename to save memory-mapped array to.  (Note: final filename will have shape info in it, and will be returned)

        order: string
            whether to save the file in 'C' or 'F' order

    Returns:
    -------
        fname_tot: the final filename of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of f
    """
    if isinstance(base_filename, list):
        raise ValueError("List of filenames is no longer supported.  save_memmap() now just saves a numpy array and formats the filename to store metadata.")

    fname_tot = gen_memmap_fname(base_filename=base_filename, shape=array.shape, order=order)

    mmap_array = np.memmap(fname_tot, mode='w+', dtype=array.dtype, shape=array.shape, order=order)
    curr_row = 0
    for tmp in np.array_split(array, n_chunks, axis=0):
        mmap_array[curr_row:curr_row + tmp.shape[0], :, :] = np.asarray(tmp, dtype=np.float32)
        mmap_array.flush()
        curr_row += tmp.shape[0]
    del mmap_array

    return fname_tot


def save_memmap_each(*args, **kwargs):
    raise DeprecationWarning(
        "save_memmap_each() no longer available. Please call save_memmap() directly instead on each array.")

def save_memmap_join(*args, **kwargs):
    raise DeprecationWarning(
        "save_memmap_join() no longer available. Please call np.concatenate() on arrays to join, then save_memmap() directly instead.")

def save_memmap_chunks(movie, base_filename, order='F', n_chunks=1):
    raise DeprecationWarning("save_memmap_chunks() no longer available. Please see save_memmap() or Movie.to_memmap() for alternative uses.")
