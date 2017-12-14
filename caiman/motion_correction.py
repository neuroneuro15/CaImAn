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
    log_xm_y, log_xp_y, log_x_ym, log_x_yp, log_xy = np.log(img[(x-1, x+1, x, x, x), (y, y, y-1, y+1, y)])
    dx = .5 * (log_xp_y - log_xm_y) / (log_xm_y + log_xp_y - 2 * log_xy)
    dy = .5 * (log_x_yp - log_x_ym) / (log_x_ym + log_x_yp - 2 * log_xy)
    return dx, dy


def bin_median(movie, window=10):
    """Returns median image of the frames of a movie after finding the mean bins of window lenght 'window'."""
    return np.median(np.mean(np.array_split(movie, window // movie.shape[0] + 1, axis=1), axis=0))


def compute_motion_shift_between_frames(img, template, max_shift_w=10, max_shift_h=10):
    """Returns (x, y) distance between an image and a template, in pixels."""
    h_i, w_i = template.shape
    templ_crop = template[max_shift_h:(h_i - max_shift_h), max_shift_w:(w_i - max_shift_w)].astype(np.float32)

    res = cv2.matchTemplate(img, templ_crop, cv2.TM_CCORR_NORMED)  # note: may want to also provide shift quality metric (ex: res.max())
    sh_y, sh_x = cv2.minMaxLoc(res)[3]
    sh_x_n, sh_y_n = max_shift_h - sh_x, max_shift_w - sh_y
    if (0 < sh_x < 2 * max_shift_h - 1) & (0 < sh_y < 2 * max_shift_w - 1):
        # if max is internal, check for subpixel shift using gaussian peak registration
        dx, dy = compute_subpixel_shift(res, sh_x_n, sh_y_n)
        sh_x_n, sh_y_n = sh_x_n + dx, sh_y_n + dy
    return sh_x_n, sh_y_n


def apply_shift(img, dx, dy, border_nan=False, border_type=cv2.BORDER_REFLECT):
    """Shifts an image by dx, dy.  This value is usually calculated from compute_motion_shift_between_frames()."""
    M = np.float32([[1, 0, dy], [0, 1, dx]])
    warped_img = cv2.warpAffine(img, M, img.shape, flags=cv2.INTER_CUBIC, borderMode=border_type)
    warped_img[:] = np.clip(warped_img, img.min(), img.max())

    if border_nan:
        max_h, max_w = np.ceil(np.maximum((0, 0), (dx, dy))).astype(np.int)
        min_h, min_w = np.floor(np.minimum((0, 0), (dx, dy))).astype(np.int)

        # todo: check logic here--it looks like some circumstances can result in all-nan images.
        warped_img[:max_h, :] = np.nan
        if min_h < 0:
            warped_img[min_h:, :] = np.nan

        warped_img[:, :max_w] = np.nan
        if min_w < 0:
            warped_img[:, min_w:] = np.nan

    return warped_img
