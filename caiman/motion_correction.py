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

from past.utils import old_div
import gc
import collections
import warnings
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2


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


def motion_correct_online(movie_iterable, add_to_movie, n_iter=1, max_shift_w=25, max_shift_h=25, save_base_name=None, order='C',
                          init_frames_template=100, show_movie=False,
                          bilateral_blur=False, diameter=10, sigmaColor=10000, sigmaSpace=0,
                          template=None, border_to_0=0, remove_blanks=False,
                          show_template=False, return_mov=False, use_median_as_template = False):
    # todo todocument

    shifts, xcorrs = [], []  # store the amount of shift in each frame
    if remove_blanks and n_iter == 1:
        raise ValueError('In order to remove blanks you need at least two iterations n_iter=2')

    if 'tifffile' in str(type(movie_iterable[0])):   
        if len(movie_iterable)==1:
            warnings.warn('NEED TO LOAD IN MEMORY SINCE SHAPE OF PAGE IS THE FULL MOVIE')
            movie_iterable = movie_iterable.asarray()
            init_mov=movie_iterable[:init_frames_template]
        else:
            init_mov=[m.asarray() for m in movie_iterable[:init_frames_template]]
    else:
        init_mov = movie_iterable[slice(0, init_frames_template, 1)]

    dims = (len(movie_iterable),) + movie_iterable[0].shape

    if use_median_as_template:
        template = bin_median(movie_iterable)

    if template is None:        
        template = (bin_median(init_mov) + add_to_movie).astype(np.float32)

    if np.percentile(template, 1) < - 10:
        raise ValueError('Movie too negative, You need to add a larger value to the Movie (add_to_movie)')

    buffer_frames, buffer_templates = collections.deque(maxlen=100), collections.deque(maxlen=100)
    max_w, max_h, min_w, min_h = 0, 0, 0, 0
    big_mov, mov = None, []
    for n in range(n_iter):

        if (save_base_name is not None) and not return_mov and (n_iter == (n+1)):

            if remove_blanks:
                dims = (dims[0], (dims[1] + min_h - max_h), (dims[2] + min_w - max_w))

            fname_tot = save_base_name + '_d1_' + str(dims[1]) + '_d2_' + str(dims[2]) + '_d3_' + str(
                1 if len(dims) == 3 else dims[3]) + '_order_' + str(order) + '_frames_' + str(dims[0]) + '_.mmap'
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=(np.prod(dims[1:]), dims[0]), order=order)

        else:
            fname_tot = None

        count = init_frames_template - 1
        shifts_tmp, xcorr_tmp = [], []
        for idx_frame, page in tqdm(enumerate(movie_iterable)):
            count += 1

            if 'tifffile' in str(type(movie_iterable[0])):
                page=page.asarray()

            img=np.array(page,dtype=np.float32)
            img=img+add_to_movie

            if bilateral_blur:
                img = compute_bilateral_blur(img, diameter, sigmaColor, sigmaSpace)

            new_img, shift, avg_corr = motion_correct_iteration_fast(img, template=template, max_shift_w=max_shift_w, max_shift_h=max_shift_h)
            template_tmp = template * count / (count + 1) + 1. / (count + 1) * new_img

            max_h,max_w = np.ceil(np.maximum((max_h,max_w),shift)).astype(np.int)
            min_h,min_w = np.floor(np.minimum((min_h,min_w),shift)).astype(np.int)

            if count < (100 + init_frames_template):
                template_old=template
                template = template_tmp
            else:
                template_old=template
            buffer_frames.append(new_img)

            if count % 100 == 0:
                if count >= (100 + init_frames_template):
                    buffer_templates.append(np.mean(buffer_frames,0))                     
                    template = np.median(buffer_templates,0)

                if show_template:
                    plt.cla()
                    plt.imshow(template,cmap='gray',vmin=250,vmax=350,interpolation='none')
                    plt.pause(.001)

            if border_to_0 > 0:
                new_img[:border_to_0,:] = 0
                new_img[:,:border_to_0] = 0
                new_img[:,-border_to_0:] = 0
                new_img[-border_to_0:,:] = 0

            shifts_tmp.append(shift)
            xcorr_tmp.append(avg_corr)

            if remove_blanks and n > 0 and (n_iter == (n+1)):
                new_img = new_img[max_h:(new_img.shape[0] - abs(min_h)), max_w:(new_img.shape[1] - abs(min_w))]

            if (save_base_name is not None) and (n_iter == (n+1)):
                big_mov[:, idx_frame] = np.reshape(new_img,np.prod(dims[1:]),order='F')

            if return_mov and (n_iter == (n+1)):
                mov.append(new_img)

            if show_movie:
                cv2.imshow('frame', old_div(new_img,500))
                print(shift)
                if not np.any(np.remainder(shift, 1) == (0, 0)):
                    cv2.waitKey(int(1./500*1000))

        shifts.append(shifts_tmp)
        xcorrs.append(xcorr_tmp)

    if save_base_name is not None:
        print('Flushing memory')
        big_mov.flush()
        del big_mov     
        gc.collect()

    if mov is not None:
        mov = np.dstack(mov).transpose([2,0,1]) 
        
    return shifts, xcorrs, template, fname_tot, mov


def motion_correct_iteration_fast(img, template, max_shift_w=10, max_shift_h=10):
    """ For using in online realtime scenarios """
    h_i, w_i = template.shape
    templ_crop = template[max_shift_h:(h_i-max_shift_h), max_shift_w:(w_i-max_shift_w)].astype(np.float32)

    res = cv2.matchTemplate(img, templ_crop, cv2.TM_CCORR_NORMED)
    sh_y, sh_x = cv2.minMaxLoc(res)[3]
    sh_x_n, sh_y_n = max_shift_h - sh_x, max_shift_w - sh_y
    if (0 < sh_x < 2 * max_shift_h - 1) & (0 < sh_y < 2 * max_shift_w - 1):
        # if max is internal, check for subpixel shift using gaussian peak registration
        dx, dy = compute_subpixel_shift(res, sh_x_n, sh_y_n)
        sh_x_n, sh_y_n = sh_x_n + dx, sh_y_n + dy

    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    new_img = cv2.warpAffine(img, M, (w_i, h_i), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    new_img[:] = np.clip(new_img, img.min(), img.max())
    shift = (sh_x_n, sh_y_n)
    avg_corr = np.max(res)
    return new_img, shift, avg_corr
