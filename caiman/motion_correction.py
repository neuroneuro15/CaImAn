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

import numpy as np
from scipy import stats
import cv2
from tqdm import tqdm


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


def low_pass_filter(img, gSig_filt):
    filt = gSig_filt[0]
    ker = cv2.getGaussianKernel((3 * filt) // 2 * 2 + 1, filt)
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


def calculate_offset(img, template, max_shift_w=10, max_shift_h=10):
    """Returns (x, y) distance between an image and a template, in pixels."""
    templ_crop = template[max_shift_h:-max_shift_h, max_shift_w:-max_shift_w]

    res = cv2.matchTemplate(img, templ_crop, cv2.TM_CCORR_NORMED)  # note: may want to also provide shift quality metric (ex: res.max())
    sh_y, sh_x = np.unravel_index(res.argmax(), res.shape)
    dx, dy = compute_subpixel_shift(res, sh_x, sh_y)
    return sh_x + dx, sh_y + dy


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