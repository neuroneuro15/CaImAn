from __future__ import division, print_function, absolute_import

import numpy as np

from caiman.io.mmapping import load_memmap


def get_patches_from_image(img, shapes, overlaps):
    # todo todocument
    _, coords_2d = extract_patch_coordinates(*np.shape(img), rf=np.divide(shapes, 2), stride=overlaps)
    imgs = img[coords_2d[:, 0], coords_2d[:, 1]].astype(np.object)
    return imgs, coords_2d


def extract_patch_coordinates(dims, rf, stride, border_pix=0):
    """
    Partitions the FOV in patches and return the indexed in 2D and 1D (flatten, order='F') formats.

    Parameters:
    ----------
    dims: tuple of int
        dimensions of the original matrix that will be  divided in patches

    rf: tuple of int
        radius of receptive field, corresponds to half the size of the square patch

    stride: tuple of int
        degree of overlap of the patches
    """
    if border_pix > 2:
        raise ValueError("border_pix must be set to 0 for 3D data since border removal is not implemented")

    dims_large = dims
    dims = np.array(dims) - border_pix * 2
    iters = [list(range(r, d - r + 1, 2 * r - s)) for r, d, s in zip(rf, dims, stride)]

    shapes, coords_flat = [], []
    coords = np.empty(list(map(len, iters)) + [len(dims)], dtype=np.object)
    for ii, xx in enumerate(iters[0]):
        coords_x = np.arange(xx - rf[0], xx + rf[0] + 1)
        coords_x = coords_x[(coords_x >= 0) & (coords_x < dims[0])]
        coords_x += border_pix

        for jj, yy in enumerate(iters[1]):
            coords_y = np.arange(yy - rf[1], yy + rf[1] + 1)
            coords_y = coords_y[(coords_y >= 0) & (coords_y < dims[1])]
            coords_y += border_pix

            if len(dims) == 2:
                idxs = np.meshgrid(coords_x, coords_y)

                coords[ii, jj] = idxs
                shapes.append(idxs[0].shape[::-1])
                coords_flat.append(np.ravel_multi_index(idxs, dims_large, order='F').flatten())

            else:  # 3D data
                for kk, zz in enumerate(iters[2]):
                    coords_z = np.arange(zz - rf[2], zz + rf[2] + 1)
                    coords_z = coords_z[(coords_z >= 0) & (coords_z < dims[2])]

                    idxs = np.meshgrid(coords_x, coords_y, coords_z)

                    shps = idxs[0].shape
                    shapes.append([shps[1], shps[0], shps[2]])
                    coords[ii, jj, kk] = idxs
                    coords_flat.append(np.ravel_multi_index(idxs, dims, order='F').flatten())

    return map(np.sort, coords_flat), shapes


def apply_to_patch(mmap_file, shape, dview, rf, stride, function, *args_in, **kwargs):
    """
    apply function to patches in parallel or not

    Parameters:
    ----------
    file_name: string
        full path to an npy file (2D, pixels x time) containing the Movie

    shape: tuple of three elements
        dimensions of the original Movie across y, x, and time


    rf: int
        half-size of the square patch in pixel

    stride: int
        amount of overlap between patches


    dview: ipyparallel view on client
        if None

    Returns:
    -------
    results

    Raise:
    -----
    Exception('Something went wrong')

    """
    T, d1, d2 = shape
    rf1, rf2 = (rf, rf) if np.isscalar(rf) else rf
    stride1, stride2 = (stride, stride) if np.isscalar(stride) else stride

    idx_flat, idx_2d = extract_patch_coordinates(dims=(d1, d2), rf=(rf1, rf2), stride=(stride1, stride2))

    # todo: simplify shape_grid assignment.  This series of conditionals is a bit confusing.
    shape_grid = tuple(np.ceil((d1 * 1. / (rf1 * 2 - stride1), d2 * 1. / (rf2 * 2 - stride2))).astype(np.int))
    if d1 <= rf1 * 2:
        shape_grid = (1, shape_grid[1])
    if d2 <= rf2 * 2:
        shape_grid = (shape_grid[0], 1)

    args_in = [(mmap_file.filename, id_f, id_2d, function, args_in, kwargs) for id_f, id_2d in
               zip(idx_flat[:], idx_2d[:])]
    if dview is not None:
        file_res = dview.map_sync(function_place_holder, args_in)
        dview.results.clear()
    else:
        file_res = list(map(function_place_holder, args_in))

    return file_res, idx_flat, shape_grid


def function_place_holder(args_in):
    # todo: todocument

    file_name, idx_, shapes, function, args, kwargs = args_in
    Yr, _, _ = load_memmap(file_name)
    Yr = Yr[idx_, :]
    Yr.filename = file_name
    d, T = Yr.shape
    Y = np.reshape(Yr, (shapes[1], shapes[0], T), order='F').transpose([2, 0, 1])
    [T, d1, d2] = Y.shape

    res_fun = function(Y, *args, **kwargs)
    if type(res_fun) is not tuple:

        if res_fun.shape == (d1, d2):
            print('** reshaping form 2D to 1D')
            res_fun = np.reshape(res_fun, d1 * d2, order='F')

    return res_fun


#
# def extract_rois_patch(file_name,d1,d2,rf=5,stride = 5):
#
#     #todo: todocument
#
#     idx_flat,idx_2d=extract_patch_coordinates(d1, d2, rf=rf,stride = stride)
#     perctl=95
#     n_components=2
#     tol=1e-6
#     max_iter=5000
#     args_in=[]
#     for id_f,id_2d in zip(idx_flat[:],idx_2d[:]):
#         args_in.append((file_name, id_f,id_2d, perctl,n_components,tol,max_iter))
#     st=time.time()
#     print((len(idx_flat)))
#     try:
#         c = Client()
#         dview=c[:]
#         file_res = dview.map_sync(nmf_patches, args_in)
#     except:
#             file_res = list(map(nmf_patches, args_in))
#     finally:
#         dview.results.clear()
#         c.purge_results('all')
#         c.purge_everything()
#         c.close()
#
#     print((time.time()-st))
#
#     A1=lil_matrix((d1*d2,len(file_res)))
#     C1=[]
#     A2=lil_matrix((d1*d2,len(file_res)))
#     C2=[]
#     for count,f in enumerate(file_res):
#         idx_,flt,ca,d=f
#         A1[idx_,count]=flt[:,0][:,np.newaxis]
#         A2[idx_,count]=flt[:,1][:,np.newaxis]
#         C1.append(ca[0,:])
#         C2.append(ca[1,:])
#
#     return A1,A2,C1,C2
