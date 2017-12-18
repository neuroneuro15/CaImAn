## -*- coding: utf-8 -*-
"""
@author Andrea Giovannucci,

The functions apply_shifts_dft, register_translation, _compute_error, _compute_phasediff, and _upsampled_dft are from 
SIMA (https://github.com/losonczylab/sima), licensed under the  GNU GENERAL PUBLIC LICENSE, Version 2, 1991. 
These same functions were adapted from sckikit-image, licensed as follows:

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
from __future__ import division, print_function, absolute_import

import itertools
import numpy as np
from scipy import stats
from tqdm import tqdm
import cv2
from .utils.stats import compute_phasediff


def guided_filter_blur_2D(movie, guide_filter, radius=5, eps=0):
    """Returns a guided-filtered version of a 3D movie array using OpenCV's ximgproc.guidedFilter()."""
    mov = movie.copy()
    for frame in tqdm(mov):
        frame[:] = cv2.ximgproc.guidedFilter(guide_filter, frame, radius=radius, eps=eps)
    return mov


def bilateral_blur_2D(movie, diameter=5, sigmaColor=10000, sigmaSpace=0):
    """Returns a bilaterally-filtered version of a 3D movie array  using openCV's bilateralFilter() function."""
    mov = movie.astype(np.float32)
    for frame in tqdm(mov):
        frame[:] = cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
    return mov


def gaussian_blur_2D(movie, kernel_size_x=5, kernel_size_y=5, kernel_std_x=1, kernel_std_y=1,
                     borderType=cv2.BORDER_REPLICATE):
    """Returns a gaussian-blurred version of a 3D movie array  using openCV's GaussianBlur() function."""
    mov = movie.copy()
    for frame in tqdm(mov):
        frame[:] = cv2.GaussianBlur(frame, ksize=(kernel_size_x, kernel_size_y), sigmaX=kernel_std_x,
                                    sigmaY=kernel_std_y, borderType=borderType)
    return mov


def median_blur_2D(movie, kernel_size=3):
    """Returns a meduian-blurred version of a 3D movie array using openCV's medianBlur() function."""
    mov = movie.copy()
    for frame in tqdm(mov):
        frame[:] = cv2.medianBlur(frame, ksize=kernel_size)
    return mov


def compute_bilateral_blur(img, diameter=10, sigmaColor=10000, sigmaSpace=0):
    return cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)


def compute_flow(frame, templ, pyr_scale=.5, levels=3, winsize=100, iterations=15, poly_n=5, poly_sigma=1.2/5, flags=0):
    return cv2.calcOpticalFlowFarneback(templ, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)


def compute_flow_normed(*args, **kwargs):
    return np.linalg.norm(compute_flow(*args, **kwargs))


def compute_correlations_with_template(movie, template):
    """Returns array of correlations between each frame of 'movie' with a same-shape template image."""
    return np.array([stats.pearsonr(frame.flatten(), template.flatten())[0] for frame in tqdm(movie)])


def compute_smoothness(movie):
    return np.sqrt(np.sum(np.power(np.gradient(np.mean(movie, axis=0)), 2)))


def compute_subpixel_shift(img, x, y):
    """takes a 2D array 'img' about index [x, y]', to check for subpixel shift using gaussian peak registration."""
    log_xm_y, log_xp_y, log_x_ym, log_x_yp, log_xy = np.log(img[(y, y, y-1, y+1, y), (x-1, x+1, x, x, x)])
    dx = .5 * (log_xp_y - log_xm_y) / (log_xm_y + log_xp_y - 2 * log_xy)
    dy = .5 * (log_x_yp - log_x_ym) / (log_x_ym + log_x_yp - 2 * log_xy)
    return dx, dy


def bin_median(movie, bins=10):
    """Returns median image of the binned meaned frames of a movie."""
    return np.median(np.mean(np.array_split(movie, bins, axis=0), axis=1), axis=0)


def low_pass_filter(img, sigma=5.):
    ker = cv2.getGaussianKernel((3 * sigma) // 2 * 2 + 1, sigma)
    ker2D = np.dot(ker, ker.T)
    ker2D[ker2D < np.max(ker2D[:,0])] = 0
    ker2D[ker2D != 0] -= np.mean(ker2D[ker2D != 0])
    return cv2.filter2D(np.array(img, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)


def dft(img):
    """Returns a frequency-space-transformed image using the discrete fourier transform."""
    freq_img = np.dstack([np.real(img), np.imag(img)])
    freq_img = cv2.dft(freq_img, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
    freq_img = np.array(freq_img[:, :, 0] + 1j * freq_img[:, :, 1], dtype=np.complex128, copy=False)
    return freq_img


def idft(freq_img):
    """Returns an image from a frequency-space-transformed image using the inverse discrete fourier transform."""
    img = cv2.idft(np.dstack([np.real(freq_img), np.imag(freq_img)]))[:, :, 0]
    return img


def calculate_offset(img, template, max_shift_w=10, max_shift_h=10, check_subpixel_shift=True, method=cv2.TM_CCORR_NORMED):
    """Returns (x, y) distance between an image and a template, in pixels."""
    templ_crop = template[max_shift_h:-max_shift_h, max_shift_w:-max_shift_w]

    res = cv2.matchTemplate(img, templ_crop, method=method)  # note: may want to also provide shift quality metric (ex: res.max())
    sh_y, sh_x = np.unravel_index(res.argmax(), res.shape)
    # return sh_y, sh_x
    if check_subpixel_shift:
        try:
            dx, dy = compute_subpixel_shift(res, sh_x, sh_y)
            sh_x, sh_y = sh_x + dx, sh_y + dy
        except IndexError:
            pass
    return sh_y - max_shift_h, sh_x - max_shift_w


def make_border_nan(img, y, x):
    """Replace a border of the image with  NaNs, depending on """
    new_img = img.copy()
    max_h, max_w = np.ceil(np.maximum((0, 0), (y, x))).astype(np.int)
    min_h, min_w = np.floor(np.minimum((0, 0), (y, x))).astype(np.int)
    new_img[:max_h, :] = np.nan
    if min_h < 0:
        new_img[min_h:, :] = np.nan
    new_img[:, :max_w] = np.nan
    if min_w < 0:
        new_img[:, min_w:] = np.nan
    return new_img


def apply_shift(img, dx, dy, border_type=cv2.BORDER_REFLECT):
    """Shifts an image by dx, dy.  This value is usually calculated from calculate_offset()."""
    M = np.float32([[1, 0, dy], [0, 1, dx]])
    warped_img = cv2.warpAffine(img, M, img.shape, flags=cv2.INTER_CUBIC, borderMode=border_type)
    warped_img[:] = np.clip(warped_img, img.min(), img.max())
    return warped_img


def apply_shift_dft(img, dx, dy, diffphase):
    """
    apply shifts using inverse dft

    src_freq: ndarray
        if is_freq it is fourier transform image else original image
    shifts: shifts to apply
    diffphase: comes from the register_translation output
    """
    freq_img = dft(img)
    nc, nr = np.shape(freq_img)
    Nr, Nc = np.meshgrid(np.fft.ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.))),
                         np.fft.ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.))))
    Greg = np.dot(freq_img * np.exp(1j * 2 * np.pi * (-dy * 1. * Nr / nr - dx * 1. * Nc / nc)), np.exp(1j * diffphase))
    shifted_img = idft(Greg)
    return shifted_img


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



def register_translation(src_image, target_image, shifts_lb=None, shifts_ub=None, max_shifts=(10,10)):
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
    shifts = maxima

    return shifts


def tile_and_correct(img, template, strides, overlaps, max_shifts, max_deviation_rigid=2, shifts_opencv=False, border_nan=True):
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
    rigid_shifts = register_translation(img, template, max_shifts=max_shifts)

    # extract patches
    sliding_img = list(sliding_window(img, overlaps=overlaps, strides=strides))
    imgs       = [it[-1] for it in sliding_img]
    start_step = [it[:2] for it in sliding_img]

    sliding_template = list(sliding_window(template, overlaps=overlaps, strides=strides))
    templates = [it[-1] for it in sliding_template]

    #extract shifts for each patch
    lb_shifts = np.ceil(np.subtract(rigid_shifts, max_deviation_rigid)).astype(int) if max_deviation_rigid is not None else None
    ub_shifts = np.floor(np.add(rigid_shifts, max_deviation_rigid)).astype(int) if max_deviation_rigid is not None else None
    shfts = [register_translation(im, template, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts) for im, template in zip(imgs, templates)]

    # create a vector field
    dim_grid = np.subtract(template.shape, np.add(overlaps, strides))
    shift_img_x, shift_img_y = np.reshape(shfts[:, 0], dim_grid), np.reshape(shfts[:, 1], dim_grid)

    dim_new_grid = np.subtract(img.shape, np.add(overlaps, strides))[::-1]
    for array in (shift_img_x, shift_img_y):
        array[:] = cv2.resize(array, dim_new_grid, interpolation=cv2.INTER_CUBIC)

    num_tiles = np.prod(dim_new_grid)
    total_shifts = [(-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]

    if shifts_opencv:
        imgs = [it[-1] for it in sliding_window(img, overlaps=overlaps, strides = strides)]
        imgs = [apply_shift(im, *sh) for im, sh in zip(imgs, total_shifts)]
    else:
        imgs = [apply_shift_dft(im, *sh) for im, sh in zip(imgs, total_shifts)]

    if border_nan:
        imgs = [make_border_nan(img, *sh) for im, sh in zip(imgs, total_shifts)]

    normalizer, new_img = np.zeros_like(img), np.zeros_like(img)
    weight_matrix = create_weight_matrix_for_blending(img, overlaps, strides)
    newshapes = np.add(strides, overlaps)
    if np.percentile([np.max(np.abs(np.diff(im, axis=axis))) for im, axis in itertools.product([shift_img_x, shift_img_y], [0, 1])], 75) < 0.5:  # calculate max_shear
        for (x, y), im, weight_mat in zip(start_step, imgs, weight_matrix):
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

