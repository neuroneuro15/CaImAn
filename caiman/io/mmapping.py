# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:33:35 2016

@author: agiovann
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from os import path


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

    fname, ext = path.splitext(base_filename)
    fname_tot = fname + '_' + order + '_' + '_'.join(map(str, array.shape))
    fname_tot = fname_tot + ext if ext else fname_tot + '.mmap_caiman'

    mmap_array = np.memmap(fname_tot, mode='w+', dtype=array.dtype, shape=array.shape, order=order)
    curr_row = 0
    for tmp in np.array_split(self, n_chunks, axis=0):
        mmap_array[curr_row:curr_row + tmp.shape[0], :, :] = np.asarray(tmp, dtype=np.float32)
        mmap_array.flush()
        curr_row += tmp.shape[0]
    del mmap_array

    return fname_tot


def save_memmap_chunks(movie, base_filename, order='F', n_chunks=1):
    raise DeprecationWarning("save_memmap_chunks() no longer available. Please see save_memmap() or Movie.to_memmap() for alternative uses.")


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

