from __future__ import division, print_function, absolute_import

import numpy as np
from .io import load_memmap

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