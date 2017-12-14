from __future__ import division, print_function, absolute_import

from past.utils import old_div
import itertools
import numpy as np
import cv2
from .utils.stats import compute_phasediff
from .motion_correction import apply_shift, make_border_nan

opencv = True


def low_pass_filter(img, gSig_filt):
    filt = gSig_filt[0]
    ker = cv2.getGaussianKernel((3 * filt) // 2 * 2 + 1, filt)
    ker2D = np.dot(ker, ker.T)
    ker2D[ker2D < np.max(ker2D[:,0])] = 0
    ker2D[ker2D != 0] -= np.mean(ker2D[ker2D != 0])
    return cv2.filter2D(np.array(img, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)



def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

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
    range_1 = list(range(0, image.shape[0]-windowSize[0], strides[0])) + [image.shape[0]-windowSize[0]]
    range_2 = list(range(0, image.shape[1]-windowSize[1], strides[1])) + [image.shape[1]-windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2,y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2 , x, y, image[ x:x + windowSize[0],y:y + windowSize[1]])


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
    for grid_1, grid_2 , x, y, _ in sliding_window(img, overlaps, strides):

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
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

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
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (np.fft.ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(old_div(data.shape[1], 2))).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            np.fft.ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(old_div(data.shape[0], 2)))
    )


    return row_kernel.dot(data).dot(col_kernel)



def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True):
    """
    apply shifts using inverse dft

    src_freq: ndarray
        if is_freq it is fourier transform image else original image
    shifts: shifts to apply
    diffphase: comes from the register_translation output


    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    shifts = shifts[::-1]

    if not is_freq:
        src_freq = np.dstack([np.real(src_freq),np.imag(src_freq)])
        src_freq = cv2.dft(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq = src_freq[:,:,0]+1j*src_freq[:, :, 1]
        src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    nc, nr = np.shape(src_freq)
    Nr, Nc = np.meshgrid(np.fft.ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.))),
                         np.fft.ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.))))

    Greg = src_freq*np.exp(1j*2*np.pi*(-shifts[0]*1.*Nr/nr-shifts[1]*1.*Nc/nc))
    Greg = Greg.dot(np.exp(1j*diffphase))
    Greg = np.dstack([np.real(Greg),np.imag(Greg)])
    new_img = cv2.idft(Greg)[:, :, 0]

    return new_img


def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb = None, shifts_ub = None, max_shifts = (10,10)):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
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
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image

    # real data needs to be fft'd.
    elif space.lower() == 'real':
        if opencv:
            src_freq_1 = cv2.dft(src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq  = src_freq_1[:,:,0]+1j*src_freq_1[:,:,1]
            src_freq   = np.array(src_freq, dtype=np.complex128, copy=False)
            target_freq_1 = cv2.dft(target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            target_freq  = target_freq_1[:,:,0]+1j*target_freq_1[:,:,1]
            target_freq = np.array(target_freq , dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = cv2.dft(target_image_cpx)

    else:
        raise ValueError("register_translation() only knows the 'real' and 'fourier' values for the 'space' argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if opencv:
        image_product_cv = np.dstack([np.real(image_product), np.imag(image_product)])
        cross_correlation = cv2.dft(image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,:,0] + 1j * cross_correlation[:,:,1]
    else:
        cross_correlation = cv2.idft(image_product)

    # Locate maximum
    new_cross_corr  = np.abs(cross_correlation)
    if (shifts_lb is not None) or (shifts_ub is not None):

        if  shifts_lb[0] < 0 and shifts_ub[0] >= 0:
            new_cross_corr[shifts_ub[0]:shifts_lb[0],:] = 0
        else:
            new_cross_corr[:shifts_lb[0],:] = 0
            new_cross_corr[shifts_ub[0]:,:] = 0

        if shifts_lb[1] < 0 and shifts_ub[1] >= 0:
            new_cross_corr[:,shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:,:shifts_lb[1]] = 0
            new_cross_corr[:,shifts_ub[1]:] = 0
    else:
        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0
        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr), cross_correlation.shape)
    shifts = np.array(maxima, dtype=np.float64)
    midpoints = np.array([axis_size // 2. for axis_size in shape])
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor > 1:  # If upsampling > 1, then refine estimate with matrix multiply DFT
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)

        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)

        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(), upsampled_region_size, upsample_factor, sample_region_offset).conj() / normalization

        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape), dtype=np.float64) - dftshift
        shifts += maxima / upsample_factor

    CCmax = cross_correlation.max()

    # If its only one row or column the shift along that dimension has no effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    phase_diff = compute_phasediff(CCmax)

    return shifts, src_freq, phase_diff


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
    rigid_shifts, sfr_freq, diffphase = register_translation(img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    # extract patches
    strides = tuple(np.round(np.divide(strides, upsample_factor_grid)).astype(np.int))

    sliding_img = list(sliding_window(img, overlaps=overlaps, strides=strides))
    imgs       = [it[-1]         for it in sliding_img]
    xy_grid    = [(it[0], it[1]) for it in sliding_img]
    start_step = [(it[2], it[3]) for it in sliding_img]

    sliding_template = list(sliding_window(template, overlaps=overlaps, strides=strides))
    dim_grid = tuple(np.add(sliding_template[-1][:2], 1))
    num_tiles = np.prod(dim_grid)

    #extract shifts for each patch
    lb_shifts = np.ceil(np.subtract(rigid_shifts, max_deviation_rigid)).astype(int) if max_deviation_rigid is not None else None
    ub_shifts = np.floor(np.add(rigid_shifts, max_deviation_rigid)).astype(int) if max_deviation_rigid is not None else None
    shfts_et_all = [register_translation(im, template, upfactor, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts)[0, 2] for im, template, upfactor in zip(imgs, [el[-1] for el in sliding_template], [upsample_factor_fft] * num_tiles)]
    shfts = np.array([sshh[0] for sshh in shfts_et_all])

    # create a vector field
    shift_img_x, shift_img_y = np.reshape(shfts[:, 0], dim_grid), np.reshape(shfts[:, 1], dim_grid)
    diffs_phase_grid = np.reshape(np.array([sshh[1] for sshh in shfts_et_all]), dim_grid)

    dim_new_grid = tuple(np.add(xy_grid[-1], 1))[::-1]
    for array in (shift_img_x, shift_img_y, diffs_phase_grid):
        array[:] = cv2.resize(array, dim_new_grid, interpolation=cv2.INTER_CUBIC)

    num_tiles = np.prod(dim_new_grid)
    total_shifts = [(-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]
    total_diffs_phase = list(diffs_phase_grid.reshape(num_tiles))

    if shifts_opencv:
        imgs = [it[-1] for it in sliding_window(img, overlaps=overlaps, strides = strides)]
        imgs = [apply_shift(im, *sh) for im, sh in zip(imgs, total_shifts)]
    else:
        imgs = [apply_shifts_dft(im, sh, dffphs, is_freq=False) for im, sh, dffphs in zip(imgs, total_shifts,total_diffs_phase)]

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

