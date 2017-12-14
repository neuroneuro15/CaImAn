from __future__ import division, print_function, absolute_import

import pickle
import numpy as np
from tqdm import tqdm
from caiman.io import load_memmap


def parallel_dot_product(A, b, block_size=5000, dview=None, transpose=False, num_blocks_per_run=20):
    # todo: todocument
    """ Chunk matrix product between matrix and column vectors

    A: memory mapped ndarray
        pixels x time

    b: time x comps
    """

    def dot_place_holder(pars):
        A_name, idx_to_pass, b_, transpose = pars
        A_, _, _ = load_memmap(A_name)
        b_ = pickle.loads(b_).astype(np.float32)

        if 'sparse' in str(type(b_)) and transpose:
            outp = (b_.T.tocsc()[:, idx_to_pass].dot(A_[idx_to_pass])).T.astype(np.float32)
        elif 'sparse' in str(type(b_)):
            outp = (b_.T.dot(A_[idx_to_pass].T)).T.astype(np.float32)
        elif transpose:
            outp = A_[idx_to_pass].dot(b_[idx_to_pass]).astype(np.float32)
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
            pars.append([A. filename, idx_to_pass, b, transpose])
        if (idx + block_size) < d1:
            idx_to_pass = list(range(idx + block_size, d1))
            pars.append([A. filename, idx_to_pass, b, transpose])
    else:
        idx_to_pass = list(range(d1))
        pars.append([A.filename, idx_to_pass, b, transpose])

    print('Start product')
    b = pickle.loads(b)
    output = np.zeros((d2, np.shape(b)[-1]), dtype=np.float32) if transpose else np.zeros((d1, np.shape(b)[-1]), dtype=np.float32)
    if dview is None:
        for counts, par in enumerate(pars):
            iddx, rs = dot_place_holder(par)
            if transpose:
                output += rs
            else:
                output[iddx] = rs
    else:
        for idx in tqdm(range(0, len(pars), num_blocks_per_run)):
            results = dview.map_sync(dot_place_holder, pars[idx:idx + num_blocks_per_run])
            results = results.get(9999999) if 'multiprocessing' in str(type(dview)) else results
            for num_, (iddx, rs) in enumerate(results):
                if transpose:
                    output += rs
                else:
                    output[iddx] = rs
            if not('multiprocessing' in str(type(dview))):
                dview.clear()

    return output
