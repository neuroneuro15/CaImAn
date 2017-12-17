from __future__ import division, print_function, absolute_import

import itertools
import numpy as np
import cv2
from .utils.stats import compute_phasediff
from .motion_correction import apply_shift, apply_shift_dft, make_border_nan, dft, idft


def sliding_window(image, overlaps, strides):
    """
    efficiently and lazily slides a window across the image

    Parameters
    ----------

    img:ndarray 2D
     image that needs to be slices

    windowSize: tuple
     dimension of the patch

    strides: tuple
     stride in wach dimension

    Returns:
    -------
    iterator containing five items

    dim_1, dim_2 coordinates in the patch grid

    x, y: bottom border of the patch in the original matrix

    patch: the patch
    """
    windowSize = np.add(overlaps,strides)
    for dim_1, x in enumerate(range(0, image.shape[0] - windowSize[0] + 1, strides[0])):
        for dim_2,y in enumerate(range(0, image.shape[1] - windowSize[1] + 1, strides[1])):
            yield (dim_1, dim_2 , image[ x:x + windowSize[0],y:y + windowSize[1]])  # yield the current window


def create_weight_matrix_for_blending(img, overlaps, strides):
    """ create a matrix that is used to normalize the intersection of the stiched patches

    Parameters:
    -----------
    img: original image, ndarray

    shapes, overlaps, strides:  tuples
        shapes, overlaps and strides of the patches

    Returns:
    --------
    weight_mat: normalizing weight matrix
    """
    shapes = np.add(strides, overlaps)
    y_overlap, x_overlap = overlaps
    max_grid_1, max_grid_2 = np.max([it[:2] for it in sliding_window(img, overlaps, strides)], axis=0)
    for grid_1, grid_2 , _ in sliding_window(img, overlaps, strides):

        weight_mat = np.ones(shapes)

        if grid_1 < max_grid_1:
            weight_mat[-y_overlap, :] = np.linspace(1, 0, y_overlap)[:, np.newaxis]
        elif grid_1 > 0:
            weight_mat[:y_overlap, :] = np.linspace(0, 1, y_overlap)[:, np.newaxis]

        if grid_2 < max_grid_2:
            weight_mat[:, -x_overlap:] = weight_mat[:,-x_overlap:] * np.linspace(1, 0, x_overlap)[np.newaxis, :]
        elif grid_2 > 0:
            weight_mat[:, :x_overlap] = weight_mat[:,:x_overlap]*np.linspace(0, 1, y_overlap)[np.newaxis, :]

        yield weight_mat


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=(0, 0)):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters:
    ----------
    data : 2D ndarray
        The input data array (DFT of original data) to upsample.

    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.

    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.

    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns:
    -------
    output : 2D ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections

    def make_kernel(shape, reg, off):
        factor = (-1j * 2 * np.pi / (shape * upsample_factor))
        kernel = np.dot(np.arange(reg)[:, None] - off, np.fft.ifftshift(np.arange(shape))[None, :] - np.floor(shape // 2.))
        return np.exp(factor * kernel)

    row_kernel, col_kernel = [make_kernel(shape, reg, off) for shape, reg, off in zip(data.shape, upsampled_region_size, axis_offsets)]

    return row_kernel.dot(data).dot(col_kernel.T)



def register_translation(src_image, target_image, upsample_factor=1, shifts_lb=None, shifts_ub=None, max_shifts=(10,10)):
    """
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters:
    ----------
    src_image : ndarray
        Reference image.

    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.

    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)

    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.

    Returns:
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.

    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    Raise:
    ------
     NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

     ValueError("Error: images must really be same size for "
                         "register_translation")

     ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    References:
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("src_image and target_image must be the same size.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    src_freq, target_freq = dft(src_image), dft(target_image)
    image_product = idft(src_freq * target_freq.conj())
    cross_correlation = idft(image_product)

    # Locate maximum
    new_cross_corr  = np.abs(cross_correlation)
    if (shifts_lb is not None) or (shifts_ub is not None):
        for lb, ub in zip(shifts_lb, shifts_ub):
            if  lb < 0 and ub >= 0:
                new_cross_corr[ub:lb, :] = 0
            else:
                new_cross_corr[:lb,:] = 0
                new_cross_corr[ub:,:] = 0
    else:
        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0
        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.array(np.unravel_index(np.argmax(np.abs(new_cross_corr)), cross_correlation.shape))
    midpoints = np.floor_divide(src_freq.shape, 2)
    maxima[maxima > midpoints] -= np.array(src_freq.shape)[maxima > midpoints]

    if upsample_factor > 1:  # If upsampling > 1, then refine estimate with matrix multiply DFT
        # Initial shift estimate in upsampled grid
        shifts = np.round(maxima * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)

        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = src_freq.size * upsample_factor ** 2

        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(), upsampled_region_size, upsample_factor, sample_region_offset).conj() / normalization

        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape), dtype=np.float64)
        maxima -= dftshift
        shifts += maxima / upsample_factor

    phase_diff = compute_phasediff(cross_correlation.max())

    return shifts, phase_diff


def tile_and_correct(img, template, strides, overlaps, max_shifts, upsample_factor_grid=4, upsample_factor_fft=10,
                     max_deviation_rigid=2, shifts_opencv=False, border_nan=True):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Parameters:
    -----------
    img: ndaarray 2D
        image to correct

    template: ndarray
        reference image

    strides: tuple
        strides of the patches in which the FOV is subdivided

    overlaps: tuple
        amount of pixel overlaping between patches along each dimension

    max_shifts: tuple
        max shifts in x and y

    newstrides:tuple
        strides between patches along each dimension when upsampling the vector fields

    newoverlaps:tuple
        amount of pixel overlaping between patches along each dimension when upsampling the vector fields

    upsample_factor_grid: int
        if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

    upsample_factor_fft: int
        resolution of fractional shifts

    show_movie: boolean whether to visualize the original and corrected frame during motion correction

    max_deviation_rigid: int
        maximum deviation in shifts of each patch from the rigid shift (should not be large)

    add_to_movie: if Movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

    filt_sig_size: tuple
        standard deviation and size of gaussian filter to center filter data in case of one photon imaging data


    """

    # compute rigid shifts
    rigid_shifts, diffphase = register_translation(img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    # extract patches
    strides = tuple(np.round(np.divide(strides, upsample_factor_grid)).astype(np.int))

    sliding_img = list(sliding_window(img, overlaps=overlaps, strides=strides))
    imgs       = [it[-1] for it in sliding_img]
    start_step = [it[:2] for it in sliding_img]

    sliding_template = list(sliding_window(template, overlaps=overlaps, strides=strides))
    templates = [it[-1] for it in sliding_template]

    #extract shifts for each patch
    lb_shifts = np.ceil(np.subtract(rigid_shifts, max_deviation_rigid)).astype(int) if max_deviation_rigid is not None else None
    ub_shifts = np.floor(np.add(rigid_shifts, max_deviation_rigid)).astype(int) if max_deviation_rigid is not None else None
    shfts_et_all = [register_translation(im, template, upsample_factor_fft, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts) for im, template in zip(imgs, templates)]
    shfts = np.array([sshh[0] for sshh in shfts_et_all])

    # create a vector field
    dim_grid = np.subtract(template.shape, np.add(overlaps, strides))
    shift_img_x, shift_img_y = np.reshape(shfts[:, 0], dim_grid), np.reshape(shfts[:, 1], dim_grid)
    diffs_phase_grid = np.reshape(np.array([sshh[1] for sshh in shfts_et_all]), dim_grid)

    dim_new_grid = np.subtract(img.shape, np.add(overlaps, strides))[::-1]
    for array in (shift_img_x, shift_img_y, diffs_phase_grid):
        array[:] = cv2.resize(array, dim_new_grid, interpolation=cv2.INTER_CUBIC)

    num_tiles = np.prod(dim_new_grid)
    total_shifts = [(-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]
    total_diffs_phase = list(diffs_phase_grid.reshape(num_tiles))

    if shifts_opencv:
        imgs = [it[-1] for it in sliding_window(img, overlaps=overlaps, strides = strides)]
        imgs = [apply_shift(im, *sh) for im, sh in zip(imgs, total_shifts)]
    else:
        imgs = [apply_shift_dft(im, *sh, diffphase=dffphs) for im, sh, dffphs in zip(imgs, total_shifts, total_diffs_phase)]

    if border_nan:
        imgs = [make_border_nan(img, *sh) for im, sh in zip(imgs, total_shifts)]

    normalizer, new_img = np.zeros_like(img), np.zeros_like(img)
    weight_matrix = create_weight_matrix_for_blending(img, overlaps, strides)
    newshapes = np.add(strides, overlaps)
    if np.percentile([np.max(np.abs(np.diff(im, axis=axis))) for im, axis in itertools.product([shift_img_x, shift_img_y], [0, 1])], 75) < 0.5:  # calculate max_shear
        for (x, y), (_, _), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):
            normalizer[x:(x + newshapes[0]), y:(y + newshapes[1])] = np.nansum(np.dstack([~np.isnan(im) * 1 * weight_mat, normalizer[x:(x + newshapes[0]), y:(y + newshapes[1])]]), -1)
            new_img[x:x + newshapes[0], y:y + newshapes[1]] = np.nansum(np.dstack([im * weight_mat, new_img[x:x + newshapes[0],y:y + newshapes[1]]]), -1)
        new_img /= normalizer
    else: # in case the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlaping area
        half_overlap_x, half_overlap_y = tuple(int(el / 2) for el in overlaps)
        for (x, y), (idx_0, idx_1), im in zip(start_step, xy_grid, imgs):
            x_start = x if idx_0 == 0 else x + half_overlap_x
            y_start = y if idx_1 == 0 else y + half_overlap_y
            new_img[x_start:(x + newshapes[0]), y_start:(y + newshapes[1])] = im[x_start-x:, y_start-y:]


    return new_img, total_shifts, start_step, xy_grid

