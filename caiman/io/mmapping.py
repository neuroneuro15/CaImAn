# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:33:35 2016

@author: agiovann
"""
from __future__ import division, print_function, absolute_import

from past.builtins import basestring
from past.utils import old_div
import numpy as np
import os
from os import path
import caiman as cm
from caiman.io import tifffile


def load_memmap(filename, mode='r', in_memory=True):
    """Returns (Numpy.mmap object, shape tuple, frame count) from a file created by save_memmap()."""
    extension = path.splitext(filename)[1]
    if not 'mmap_caiman' in extension:
        fdata = path.basename(filename).split('_')[1:]
        fname, order, shape = fdata[0], fdata[1], fdata[2:]
    else:
        d1, d2, d3, order, T = [int(part) for part in path.basename(filename).split('_')[-9::2]]
        # shape = (d1, d2) if d3 == 1 else (d1, d2, d3)
        shape = (d1 * d2 * d3, T)

    Yr = np.memmap(filename, mode=mode, shape=shape, dtype=np.float32, order=order)
    Yr = np.array(Yr) if in_memory else Yr

    return Yr, shape, T


def save_memmap_each(fnames, dview=None, base_name=None, resize_fact=(1, 1, 1), remove_init=0,
                     idx_xy=None, xy_shifts=None, add_to_movie=0, border_to_0=0, order='C'):
    """
    Create several memory mapped files using parallel processing

    Parameters:
    -----------
    fnames: list of str
        list of path to the filenames

    dview: ipyparallel dview
        used to perform computation in parallel. If none it will be signle thread

    base_name str
        BaseName for the file to be creates. If not given the file itself is used

    resize_fact: tuple
        resampling factors for each dimension x,y,time. .1 = downsample 10X

    remove_init: int
        number of samples to remove from the beginning of each chunk

    idx_xy: slice operator
        used to perform slicing of the Movie (to select a subportion of the Movie)

    xy_shifts: list
        x and y shifts computed by a motion correction algorithm to be applied before memory mapping

    add_to_movie: float
        if Movie too negative will make it positive

    border_to_0: int
        number of pixels on the border to set to the minimum of the Movie

    Returns:
    --------
    fnames_tot: list
        paths to the created memory map files

    """

    def save_place_holder(pars):
        """ To use map reduce"""
        (f, base_name, resize_fact, remove_init, idx_xy, order,
         xy_shifts, add_to_movie, border_to_0) = pars
        return save_memmap([f], base_name=base_name, resize_fact=resize_fact, remove_init=remove_init,
                           idx_xy=idx_xy, order=order, xy_shifts=xy_shifts,
                           add_to_movie=add_to_movie, border_to_0=border_to_0)


    xy_shifts = [None] * len(fnames) if type(xy_shifts) == type(None) else xy_shifts
    resize_fact = [resize_fact] * len(fnames) if not isinstance(resize_fact, list) else resize_fact
    pars = []
    for idx, (f, shift, resize) in enumerate(zip(fnames, xy_shifts, resize_fact)):
        fname = '{}{:04d}'.format(base_name, idx) if not base_name else os.path.splitext(f)[0]
        par = [f, fname, resize, remove_init, idx_xy, order, shift, add_to_movie, border_to_0]
        pars.append(par)

    if 'multiprocessing' in str(type(dview)):
        fnames_new = dview.map_async(save_place_holder, pars).get(9999999)
    elif type(dview) != type(None):
        fnames_new = dview.map_sync(save_place_holder, pars)
    else:
        fnames_new = list(map(save_place_holder, pars))

    return fnames_new


#%%
def save_memmap_join(mmap_fnames, base_name=None, n_chunks=20, dview=None, order='C'):
    """
    From small memory mappable files creates a large one

    Parameters:
    -----------

    mmap_fnames: list of memory mapped files

    base_name: string, will be the first portion of name to be solved

    n_chunks: number of chunks in which to subdivide when saving, smaller requires more memory

    dview: cluster handle

    Returns:
    --------

    """

    tot_frames = 0
    for f in mmap_fnames:
        Yr, dims, T = load_memmap(f)
        tot_frames += T
        del Yr

    d = np.prod(dims)
    if base_name is None:
        base_name = mmap_fnames[0]
        base_name = base_name[:base_name.find('_d1_')] + '-#-' + str(len(mmap_fnames))

    fname_tot = (base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' +
                 str(1 if len(dims) == 2 else dims[2]) + '_order_' + str(order) +
                 '_frames_' + str(tot_frames) + '_.mmap')
    fname_tot = os.path.join(os.path.split(mmap_fnames[0])[0], fname_tot)

    print(fname_tot)

    big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=(d, tot_frames), order='C')

    step = d // n_chunks
    pars = []
    for ref in range(0, d - step + 1, step):
        pars.append([fname_tot, d, tot_frames, mmap_fnames, ref, ref + step])
    # last batch should include the leftover pixels
    pars[-1][-1] = d

    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            dview.map_async(save_portion, pars).get(9999999)            
        else:
            dview.map_sync(save_portion, pars)
    else:
        list(map(save_portion, pars))

    np.savez(base_name + '.npz', mmap_fnames=mmap_fnames, fname_tot=fname_tot)

    print('Deleting big mov')
    del big_mov

    return fname_tot


def save_portion(pars):
    # todo: todocument

    big_mov, d, tot_frames, fnames, idx_start, idx_end = pars
    big_mov = np.memmap(big_mov, mode='r+', dtype=np.float32, shape=(d, tot_frames), order='C')
    Ttot = 0
    Yr_tot = np.zeros((idx_end - idx_start, tot_frames))
    print((Yr_tot.shape))
    for f in fnames:
        print(f)
        Yr, dims, T = load_memmap(f)
        print((idx_start, idx_end))
        Yr_tot[:, Ttot:Ttot + T] = np.array(Yr[idx_start:idx_end])
        Ttot = Ttot + T
        del Yr

    big_mov[idx_start:idx_end, :] = Yr_tot
    del Yr_tot
    print('done')
    del big_mov
    return Ttot



#%%
def save_memmap(filenames, base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None,
                order='F', xy_shifts=None, is_3D=False, add_to_movie=0, border_to_0=0):
    """ Saves efficiently a list of tif files into a memory mappable file

    Parameters:
    ----------
        filenames: list
            list of tif files or list of numpy arrays

        base_name: str
            the base used to build the file name. IT MUST NOT CONTAIN "_"

        resize_fact: tuple
            x,y, and z downampling factors (0.5 means downsampled by a factor 2)

        remove_init: int
            number of frames to remove at the begining of each tif file
            (used for resonant scanning images if laser in rutned on trial by trial)

        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance
            idx_xy = (slice(150,350,None), slice(150,350,None))

        order: string
            whether to save the file in 'C' or 'F' order

        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping

        is_3D: boolean
            whether it is 3D data
    Returns:
    -------
        fname_new: the name of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of f

    """

    # TODO: can be done online
    Ttot = 0
    for idx, f in enumerate(filenames):
        if isinstance(f, str):
            print(f)

        if is_3D:
            Yr = f if isinstance(f, basestring) else tifffile.imread(f)
            if idx_xy is None:
                Yr = Yr[remove_init:]
            elif len(idx_xy) == 2:
                Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                Yr = Yr[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

        else:
            Yr = cm.load(f, fr=1, in_memory=True) if isinstance(f, basestring) else cm.Movie(f)
            if xy_shifts is not None:
                Yr = Yr.apply_motion_correction(xy_shifts, interpolation='cubic', remove_blanks=False)

            if idx_xy is None:
                if remove_init > 0:
                    Yr = np.array(Yr)[remove_init:]
            elif len(idx_xy) == 2:
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                raise Exception('You need to set is_3D=True for 3D data)')
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

        if border_to_0 > 0:

            min_mov = np.nanmin(Yr)
            Yr[:, :border_to_0, :] = min_mov
            Yr[:, :, :border_to_0] = min_mov
            Yr[:, :, -border_to_0:] = min_mov
            Yr[:, -border_to_0:, :] = min_mov

        fx, fy, fz = resize_fact
        if fx != 1 or fy != 1 or fz != 1:

            if 'Movie' not in str(type(Yr)):
                Yr = cm.Movie(Yr, fr=1)

            Yr = Yr.resize(fx=fx, fy=fy, fz=fz)
        T, dims = Yr.shape[0], Yr.shape[1:]
        Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
        Yr = np.reshape(Yr, (np.prod(dims), T), order='F')

        if idx == 0:
            fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(
                1 if len(dims) == 2 else dims[2]) + '_order_' + str(order)
            if isinstance(f, str):
                fname_tot = os.path.join(os.path.split(f)[0], fname_tot)
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                                shape=(np.prod(dims), T), order=order)
        else:
            big_mov = np.memmap(fname_tot, dtype=np.float32, mode='r+',
                                shape=(np.prod(dims), Ttot + T), order=order)

        big_mov[:, Ttot:Ttot + T] = np.asarray(Yr, dtype=np.float32) + 1e-10 + add_to_movie
        big_mov.flush()
        del big_mov
        Ttot = Ttot + T

    fname_new = fname_tot + '_frames_' + str(Ttot) + '_.mmap'
    os.rename(fname_tot, fname_new)

    return fname_new


def save_memmap_chunks(movie, base_filename, order='F', n_chunks=1):
    raise DeprecationWarning("save_memmap_chunks() no longer available. Please see save_memmap() or Movie.to_memmap() for alternative uses.")


#%%
def parallel_dot_product(A, b, block_size=5000, dview=None, transpose=False, num_blocks_per_run=20):
    # todo: todocument
    """ Chunk matrix product between matrix and column vectors

    A: memory mapped ndarray
        pixels x time

    b: time x comps
    """

    import pickle

    def dot_place_holder(par):
        A_name, idx_to_pass, b_, transpose = par
        A_, _, _ = load_memmap(A_name)
        b_ = pickle.loads(b_).astype(np.float32)

        print((idx_to_pass[-1]))
        if 'sparse' in str(type(b_)):
            if transpose:
                outp = (b_.T.tocsc()[:, idx_to_pass].dot(A_[idx_to_pass])).T.astype(np.float32)
                #            del b_
                #            return idx_to_pass, outp

            else:
                outp = (b_.T.dot(A_[idx_to_pass].T)).T.astype(np.float32)
                #            del b_
                #            return idx_to_pass,outp
        else:
            if transpose:
                outp = A_[idx_to_pass].dot(b_[idx_to_pass]).astype(np.float32)
                #            del b_
                #            return idx_to_pass, outp
            else:

                outp = A_[idx_to_pass].dot(b_).astype(np.float32)

        del b_, A_
        return idx_to_pass, outp


    pars = []
    d1, d2 = np.shape(A)
    b = pickle.dumps(b)
    print('parallel dot product block size: ' + str(block_size))

    if block_size < d1:

        for idx in range(0, d1 - block_size, block_size):
            idx_to_pass = list(range(idx, idx + block_size))
            pars.append([A.filename, idx_to_pass, b, transpose])

        if (idx + block_size) < d1:
            idx_to_pass = list(range(idx + block_size, d1))
            pars.append([A.filename, idx_to_pass, b, transpose])

    else:
        idx_to_pass = list(range(d1))
        pars.append([A.filename, idx_to_pass, b, transpose])

    print('Start product')
    b = pickle.loads(b)

    if transpose:
        output = np.zeros((d2, np.shape(b)[-1]), dtype=np.float32)
    else:
        output = np.zeros((d1, np.shape(b)[-1]), dtype=np.float32)

    if dview is None:
        if transpose:
            #            b = pickle.loads(b)
            print('Transposing')
#            output = np.zeros((d2,np.shape(b)[-1]), dtype = np.float32)
            for counts, pr in enumerate(pars):

                iddx, rs = dot_place_holder(pr)
                output = output + rs

        else:
            #            b = pickle.loads(b)
            #            output = np.zeros((d1,np.shape(b)[-1]), dtype = np.float32)
            for counts, pr in enumerate(pars):
                iddx, rs = dot_place_holder(pr)
                output[iddx] = rs

    else:
        #        b = pickle.loads(b)

        for itera in range(0, len(pars), num_blocks_per_run):
            
            if 'multiprocessing' in str(type(dview)):
                results = dview.map_async(dot_place_holder, pars[itera:itera + num_blocks_per_run]).get(9999999)
            else:
                results = dview.map_sync(dot_place_holder, pars[itera:itera + num_blocks_per_run])
                
            print('Processed:' + str([itera, itera + len(results)]))

            if transpose:
                print('Transposing')

                for num_, res in enumerate(results):
                    #                    print(num_)
                    output += res[1]

            else:
                print('Filling')

                for res in results:
                    output[res[0]] = res[1]

            if not('multiprocessing' in str(type(dview))):
                dview.clear()

    return output
