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
import itertools
import warnings
from tqdm import tqdm

import numpy as np
from numpy.fft import ifftshift
import matplotlib.pyplot as plt
import cv2

from .movie import Movie

opencv = True


def apply_shifts_movie(movie, coord_shifts_els, x_shifts_els, y_shifts_els, rigid_shifts=True, shifts_opencv=True, shifts_rig=14):
    """
    Applies shifts found by registering one file to a different file. Useful
    for cases when shifts computed from a structural channel are applied to a
    functional channel. Currently only application of shifts through openCV is
    supported.

    Parameters:
    -----------
    fname: str
        name of the Movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

    rigid_shifts: bool
        apply rigid or pw-rigid shifts (must exist in the mc object)

    Returns:
    ----------
    m_reg: caiman Movie object
        caiman Movie object with applied shifts (not memory mapped)
    """

    if rigid_shifts:
        if shifts_opencv:
            m_reg = [apply_shift_iteration(img, shift) for img, shift in zip(movie, shifts_rig)]
        else:
            m_reg = [apply_shifts_dft(img, (sh[0], sh[1]), 0, is_freq=False, border_nan=True)  for img, sh in zip(movie, shifts_rig)]
    else:
        dims_grid = tuple(np.max(np.stack(coord_shifts_els[0],axis=1),axis=1) - np.min(np.stack(coord_shifts_els[0],axis=1),axis=1) + 1)
        shifts_x = np.stack([np.reshape(_sh_,dims_grid,order='C').astype(np.float32) for _sh_ in x_shifts_els], axis = 0)
        shifts_y = np.stack([np.reshape(_sh_,dims_grid,order='C').astype(np.float32) for _sh_ in y_shifts_els], axis = 0)
        dims = movie.shape[1:]
        x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32), np.arange(0., dims[1]).astype(np.float32))
        m_reg = [cv2.remap(img,
                    -np.resize(shiftY, dims)+x_grid, -np.resize(shiftX, dims)+y_grid, cv2.INTER_CUBIC)
                    for img, shiftX, shiftY in zip(movie, shifts_x, shifts_y)]

    return Movie(np.stack(m_reg, axis=0))


def apply_shift_iteration(img,shift,border_nan=False, border_type = cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n,sh_y_n = shift
    w_i,h_i=img.shape
    M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])    
    min_,max_ = np.min(img),np.max(img)
    img = np.clip(cv2.warpAffine(img,M,(h_i,w_i),flags=cv2.INTER_CUBIC, borderMode = border_type),min_,max_)
    if border_nan:  
        max_w,max_h,min_w,min_h=0,0,0,0
        max_h,max_w = np.ceil(np.maximum((max_h,max_w),shift)).astype(np.int)
        min_h,min_w = np.floor(np.minimum((min_h,min_w),shift)).astype(np.int)
        img[:max_h,:] = np.nan
        if min_h < 0:
            img[min_h:,:] = np.nan
        img[:,:max_w] = np.nan 
        if min_w < 0:
            img[:,min_w:] = np.nan

    return img


def motion_correct_online(movie_iterable,add_to_movie,max_shift_w=25,max_shift_h=25,save_base_name=None,order = 'C',
                        init_frames_template=100, show_movie=False, bilateral_blur=False,template=None, min_count=1000,
                        border_to_0=0, n_iter = 1, remove_blanks=False,show_template=False,return_mov=False,
                        use_median_as_template = False):
    # todo todocument

    shifts=[]  # store the amount of shift in each frame
    xcorrs=[]
    if remove_blanks and n_iter==1:
        raise Exception('In order to remove blanks you need at least two iterations n_iter=2')

    if 'tifffile' in str(type(movie_iterable[0])):   
        if len(movie_iterable)==1:
            print('******** WARNING ****** NEED TO LOAD IN MEMORY SINCE SHAPE OF PAGE IS THE FULL MOVIE')
            movie_iterable = movie_iterable.asarray()
            init_mov=movie_iterable[:init_frames_template]
        else:
            init_mov=[m.asarray() for m in movie_iterable[:init_frames_template]]
    else:
        init_mov=movie_iterable[slice(0,init_frames_template,1)]

    dims=(len(movie_iterable),)+movie_iterable[0].shape 
    print(("dimensions:" + str(dims)))

    if use_median_as_template:
        template = bin_median(movie_iterable)

    if template is None:        
        template = bin_median(init_mov)
        count=init_frames_template
        if np.percentile(template, 1) + add_to_movie < - 10:
            raise Exception('Movie too negative, You need to add a larger value to the Movie (add_to_movie)')
        template=np.array(template + add_to_movie,dtype=np.float32)    
    else:
        if np.percentile(template, 1) < - 10:
            raise Exception('Movie too negative, You need to add a larger value to the Movie (add_to_movie)')
        count=min_count

    min_mov = 0
    buffer_size_frames=100          
    buffer_size_template=100     
    buffer_frames=collections.deque(maxlen=buffer_size_frames)  
    buffer_templates=collections.deque(maxlen=buffer_size_template)  
    max_w,max_h,min_w,min_h=0,0,0,0

    big_mov = None
    if return_mov:
        mov=[]   
    else:
        mov = None

    for n in range(n_iter):
        if n>0:
            count = init_frames_template

        if (save_base_name is not None) and (big_mov is None) and (n_iter == (n+1)):  

            if remove_blanks:
                dims = (dims[0],dims[1]+min_h-max_h,dims[2]+min_w-max_w)

            fname_tot = save_base_name + '_d1_' + str(dims[1]) + '_d2_' + str(dims[2]) + '_d3_' + str(
                1 if len(dims) == 3 else dims[3]) + '_order_' + str(order) + '_frames_' + str(dims[0]) + '_.mmap'
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=(np.prod(dims[1:]), dims[0]), order=order)

        else:
            fname_tot = None  

        shifts_tmp = []
        xcorr_tmp = []
        for idx_frame,page in enumerate(movie_iterable):  

            if 'tifffile' in str(type(movie_iterable[0])):
                page=page.asarray()

            img=np.array(page,dtype=np.float32)
            img=img+add_to_movie

            new_img,template_tmp,shift,avg_corr = motion_correct_iteration(
                img,template,count,max_shift_w=max_shift_w,max_shift_h=max_shift_h,bilateral_blur=bilateral_blur)

            max_h,max_w = np.ceil(np.maximum((max_h,max_w),shift)).astype(np.int)
            min_h,min_w = np.floor(np.minimum((min_h,min_w),shift)).astype(np.int)

            if count < (buffer_size_frames + init_frames_template):
                template_old=template
                template = template_tmp
            else:
                template_old=template
            buffer_frames.append(new_img)

            if count%100 == 0:
                if count >= (buffer_size_frames + init_frames_template):
                    buffer_templates.append(np.mean(buffer_frames,0))                     
                    template = np.median(buffer_templates,0)

                if show_template:
                    plt.cla()
                    plt.imshow(template,cmap='gray',vmin=250,vmax=350,interpolation='none')
                    plt.pause(.001)

                print(('Relative change in template:' + str(
                    old_div(np.sum(np.abs(template-template_old)),np.sum(np.abs(template))))))
                print(('Iteration:'+ str(count)))

            if border_to_0 > 0:
                new_img[:border_to_0,:]=min_mov
                new_img[:,:border_to_0]=min_mov
                new_img[:,-border_to_0:]=min_mov
                new_img[-border_to_0:,:]=min_mov

            shifts_tmp.append(shift)
            xcorr_tmp.append(avg_corr)

            if remove_blanks and n>0  and (n_iter == (n+1)):

                new_img = new_img[max_h:,:]
                if min_h < 0:
                    new_img = new_img[:min_h,:]
                new_img = new_img[:,max_w:] 
                if min_w < 0:
                    new_img = new_img[:,:min_w]

            if (save_base_name is not None) and (n_iter == (n+1)):

                big_mov[:,idx_frame] = np.reshape(new_img,np.prod(dims[1:]),order='F')

            if return_mov and (n_iter == (n+1)):
                mov.append(new_img)

            if show_movie:
                cv2.imshow('frame',old_div(new_img,500))
                print(shift)
                if not np.any(np.remainder(shift,1) == (0,0)):
                    cv2.waitKey(int(1./500*1000))

            count+=1
        shifts.append(shifts_tmp)
        xcorrs.append(xcorr_tmp)

    if save_base_name is not None:
        print('Flushing memory')
        big_mov.flush()
        del big_mov     
        gc.collect()

    if mov is not None:
        mov = np.dstack(mov).transpose([2,0,1]) 
        
    return shifts,xcorrs,template, fname_tot, mov


#%%
def motion_correct_iteration(img,template,frame_num,max_shift_w=25,
                             max_shift_h=25,bilateral_blur=False,diameter=10,sigmaColor=10000,sigmaSpace=0):
#todo todocument
    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w

    if bilateral_blur:
        img=cv2.bilateralFilter(img,diameter,sigmaColor,sigmaSpace)    
    templ_crop=template[max_shift_h:h_i-max_shift_h,max_shift_w:w_i-max_shift_w].astype(np.float32)
    res = cv2.matchTemplate(img,templ_crop,cv2.TM_CCORR_NORMED)

    top_left = cv2.minMaxLoc(res)[3]
    avg_corr=np.max(res)
    sh_y,sh_x = top_left

    if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
        # if max is internal, check for subpixel shift using gaussian
        # peak registration
        log_xm1_y = np.log(res[sh_x-1,sh_y])
        log_xp1_y = np.log(res[sh_x+1,sh_y])
        log_x_ym1 = np.log(res[sh_x,sh_y-1])
        log_x_yp1 = np.log(res[sh_x,sh_y+1])
        four_log_xy = 4*np.log(res[sh_x,sh_y])

        sh_x_n = -(sh_x - ms_h + old_div((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
        sh_y_n = -(sh_y - ms_w + old_div((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
    else:
        sh_x_n = -(sh_x - ms_h)
        sh_y_n = -(sh_y - ms_w)

    M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
    min_,max_ = np.min(img),np.max(img)
    new_img = np.clip(cv2.warpAffine(img,M,(w_i,h_i),flags=cv2.INTER_CUBIC, borderMode = cv2.BORDER_REFLECT),min_,max_)

    new_templ=template*frame_num/(frame_num + 1) + 1./(frame_num + 1)*new_img     
    shift=[sh_x_n,sh_y_n]

    return new_img,new_templ,shift,avg_corr

#%%
def motion_correct_iteration_fast(img,template,max_shift_w=10,max_shift_h=10):
    """ For using in online realtime scenarios """
    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w

    templ_crop=template[max_shift_h:h_i-max_shift_h,max_shift_w:w_i-max_shift_w].astype(np.float32)

    h,w = templ_crop.shape

    res = cv2.matchTemplate(img,templ_crop,cv2.TM_CCORR_NORMED)
    top_left = cv2.minMaxLoc(res)[3]

    sh_y,sh_x = top_left

    if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
        # if max is internal, check for subpixel shift using gaussian
        # peak registration
        log_xm1_y = np.log(res[sh_x-1,sh_y]);
        log_xp1_y = np.log(res[sh_x+1,sh_y]);
        log_x_ym1 = np.log(res[sh_x,sh_y-1]);
        log_x_yp1 = np.log(res[sh_x,sh_y+1]);
        four_log_xy = 4*np.log(res[sh_x,sh_y]);

        sh_x_n = -(sh_x - ms_h + old_div((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
        sh_y_n = -(sh_y - ms_w + old_div((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
    else:
        sh_x_n = -(sh_x - ms_h)
        sh_y_n = -(sh_y - ms_w)

    M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
    
    new_img = cv2.warpAffine(img,M,(w_i,h_i),flags=cv2.INTER_CUBIC, borderMode = cv2.BORDER_REFLECT)

    shift=[sh_x_n,sh_y_n]

    return new_img, shift

#%%    
def bin_median(mat,window=10,exclude_nans = False ):

    """ compute median of 3D array in along axis o by binning values

    Parameters:
    ----------

    mat: ndarray
        input 3D matrix, time along first dimension

    window: int
        number of frames in a bin


    Returns:
    -------
    img: 
        median image


    Raise:
    -----
    Exception('Path to template does not exist:'+template)
    """
    
    T,d1,d2=np.shape(mat)
    if T < window:
        window = T
    num_windows=np.int(old_div(T,window))
    num_frames=num_windows*window
    if exclude_nans:
        img=np.nanmedian(np.nanmean(np.reshape(mat[:num_frames],(window,num_windows,d1,d2)),axis=0),axis=0)    
    else:
        img=np.median(np.mean(np.reshape(mat[:num_frames],(window,num_windows,d1,d2)),axis=0),axis=0)        

    return img


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
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(old_div(data.shape[1], 2))).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(old_div(data.shape[0], 2)))
    )


    return row_kernel.dot(data).dot(col_kernel)


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Parameters:
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Parameters:
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.

    src_amp : float
        The normalized average image intensity of the source image

    target_amp : float
        The normalized average image intensity of the target image
    """
    return np.sqrt(np.abs(1.0 - cross_correlation_max * cross_correlation_max.conj() / (src_amp * target_amp)))

#%%
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
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if opencv:

        image_product_cv = np.dstack([np.real(image_product),np.imag(image_product)])
        cross_correlation = cv2.dft(image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,:,0]+1j*cross_correlation[:,:,1]
    else:
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = cv2.idft(image_product)

    # Locate maximum
    new_cross_corr  = np.abs(cross_correlation)


    if (shifts_lb is not None) or (shifts_ub is not None):

        if  (shifts_lb[0]<0) and (shifts_ub[0]>=0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0],:] = 0                                                                  
        else:
            new_cross_corr[:shifts_lb[0],:] = 0                
            new_cross_corr[shifts_ub[0]:,:] = 0    

        if  (shifts_lb[1]<0) and (shifts_ub[1]>=0):      
            new_cross_corr[:,shifts_ub[1]:shifts_lb[1]] = 0                                                      
        else:
            new_cross_corr[:,:shifts_lb[1]] = 0    
            new_cross_corr[:,shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0],:] = 0   

        new_cross_corr[:,max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2)) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0



    return shifts, src_freq,_compute_phasediff(CCmax)

#%%        
def apply_shifts_dft(src_freq, shifts, diffphase, is_freq = True, border_nan = False):
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
    apply shifts using inverse dft
    src_freq: ndarray
        if is_freq it is fourier transform image else original image
    shifts: shifts to apply
    diffphase: comes from the register_translation output

    """
    shifts = shifts[::-1]
    if not is_freq:
        src_freq = np.dstack([np.real(src_freq),np.imag(src_freq)])
        src_freq = cv2.dft(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq  = src_freq[:,:,0]+1j*src_freq[:,:,1]
        src_freq   = np.array(src_freq, dtype=np.complex128, copy=False)          

    nc,nr = np.shape(src_freq)
    Nr = ifftshift(np.arange(-np.fix(old_div(nr,2.)),np.ceil(old_div(nr,2.))))
    Nc = ifftshift(np.arange(-np.fix(old_div(nc,2.)),np.ceil(old_div(nc,2.))))
    Nr,Nc = np.meshgrid(Nr,Nc)

    Greg = src_freq*np.exp(1j*2*np.pi*(-shifts[0]*1.*Nr/nr-shifts[1]*1.*Nc/nc))
    Greg = Greg.dot(np.exp(1j*diffphase))
    Greg = np.dstack([np.real(Greg),np.imag(Greg)])
    new_img = cv2.idft(Greg)[:, :, 0]
    if border_nan:  
        max_w,max_h,min_w,min_h=0,0,0,0
        max_h,max_w = np.ceil(np.maximum((max_h,max_w),shifts)).astype(np.int)
        min_h,min_w = np.floor(np.minimum((min_h,min_w),shifts)).astype(np.int)
        new_img[:max_h,:] = np.nan
        if min_h < 0:
            new_img[min_h:,:] = np.nan
        new_img[:,:max_w] = np.nan 
        if min_w < 0:
            new_img[:,min_w:] = np.nan

    return new_img


#%%
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


def low_pass_filter_space(img_orig, gSig_filt):
    filt = gSig_filt[0]
    ker = cv2.getGaussianKernel((3 * filt) // 2 * 2 + 1, filt)
    ker2D = np.dot(ker, ker.T)
    ker2D[ker2D < np.max(ker2D[:,0])] = 0
    ker2D[ker2D != 0] -= np.mean(ker2D[ker2D != 0])
    return cv2.filter2D(np.array(img_orig, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)


def tile_and_correct(img, template, strides, overlaps, max_shifts, upsample_factor_grid=4, upsample_factor_fft=10, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None):

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


#    if (add_to_movie != 0) and gSig_filt is not None:
#        raise Exception('When gSig_filt or gSiz_filt are used add_to_movie must be zero!')

    if gSig_filt is not None and not opencv:
        raise NotImplementedError('The use of FFT and filtering options have not been tested. Set opencv=True')

    if gSig_filt is not None:
        img_orig = img.copy()
        img = low_pass_filter_space(img_orig, gSig_filt)
    else:
        img = img.astype(np.float64)

    img += add_to_movie
    template = template.astype(np.float64) + add_to_movie

    # compute rigid shifts
    rigid_shifts, sfr_freq, diffphase = register_translation(img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)
    if max_deviation_rigid == 0:
        if shifts_opencv:
            img = img_orig if gSig_filt is not None else img
            new_img = apply_shift_iteration(img, (-rigid_shifts[0], -rigid_shifts[1]), border_nan=False)
        else:
            new_img = apply_shifts_dft(sfr_freq,(-rigid_shifts[0], -rigid_shifts[1]), diffphase, border_nan=True)
        new_img -= add_to_movie
        return (new_img, (-rigid_shifts[0], -rigid_shifts[1]), None, None)

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
        if gSig_filt is not None:
            imgs = [it[-1] for it in sliding_window(img_orig, overlaps=overlaps, strides = strides)]
        imgs = [apply_shift_iteration(im,sh,border_nan=True) for im, sh in zip(imgs, total_shifts)]
    else:
        imgs = [apply_shifts_dft(im, sh, dffphs, is_freq=False, border_nan=True) for im, sh, dffphs in zip(imgs, total_shifts,total_diffs_phase)]

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

    new_img -= add_to_movie

    return new_img, total_shifts, start_step, xy_grid


def show_tile_and_correct_movie(img, new_img, template, sfr_freq, rigid_shifts, diffphase):
    """Takes the movie returned by tile_and_correct() and displays it in an OpenCV window."""
    img = apply_shifts_dft(sfr_freq, (-rigid_shifts[0], -rigid_shifts[1]), diffphase, border_nan=True)
    img_show = np.vstack([new_img, img])
    img_show = cv2.resize(img_show, None, fx=1, fy=1)
    img_show /= np.percentile(template, 99)

    cv2.imshow('frame', img_show)
    cv2.waitKey(2)
    cv2.destroyAllWindows()


def compute_flow_single_frame(frame, templ, pyr_scale=.5, levels=3, winsize=100, iterations=15, poly_n=5, poly_sigma=1.2/5, flags=0):
    return cv2.calcOpticalFlowFarneback(templ, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)


def compute_metrics_motion_correction(fname,final_size_x,final_size_y, swap_dim,pyr_scale = .5,levels = 3,
                                      winsize = 100, iterations = 15, poly_n = 5, poly_sigma = 1.2/5, flags = 0,
                                      play_flow = False, resize_fact_flow = .2,template = None):
    #todo: todocument
    #cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    import scipy
    vmin, vmax = -1, 1
    m = Movie.load(fname)

    max_shft_x = np.int(np.ceil((np.shape(m)[1]-final_size_x)/2))
    max_shft_y = np.int(np.ceil((np.shape(m)[2]-final_size_y)/2))
    max_shft_x_1 = - ( (np.shape(m)[1]-max_shft_x)-(final_size_x) )
    max_shft_y_1 = - ( (np.shape(m)[2]-max_shft_y)-(final_size_y) )
    if max_shft_x_1 == 0:
        max_shft_x_1 = None
        
    if max_shft_y_1 == 0:
        max_shft_y_1 = None
    print ([max_shft_x,max_shft_x_1,max_shft_y,max_shft_y_1])    
    m = m[:,max_shft_x:max_shft_x_1,max_shft_y:max_shft_y_1]
    if np.sum(np.isnan(m))>0:
        print(m.shape)
        raise Exception('Movie contains nan')
        
    print('Local correlations..')
    img_corr = m.local_correlations(eight_neighbours=True, swap_dim = swap_dim)
    print (m.shape)
    if template is None:
        tmpl = bin_median(m)
    else:
        tmpl = template

    
    print('Compute Smoothness.. ')
    smoothness = np.sqrt(np.sum(np.sum(np.array(np.gradient(np.mean(m,0)))**2,0)))
    smoothness_corr = np.sqrt(np.sum(np.sum(np.array(np.gradient(img_corr))**2,0)))
    
    print('Compute correlations.. ')
    correlations = []
    count = 0
    for fr in m:
         if count%100 == 0:
            print(count)   
        
         count +=1    
         correlations.append(scipy.stats.pearsonr(fr.flatten(),tmpl.flatten())[0]) 
        
    print('Compute optical flow .. ')
    
    m = m.resize(1,1,resize_fact_flow)
    norms = []
    flows = []
    count = 0
    for fr in m:
        if count%100 == 0:
            print(count)   
        
        count +=1    
        flow = cv2.calcOpticalFlowFarneback(tmpl,fr,None,pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        
        if play_flow:
            plt.subplot(1,3,1)
            plt.cla()
            plt.imshow(fr,vmin = 0, vmax = 300, cmap = 'gray' )
            plt.title('Movie')
            plt.subplot(1,3,3)
            plt.cla()
            plt.imshow(flow[:,:,1],vmin=vmin,vmax=vmax)
            plt.title('y_flow')
            
            plt.subplot(1,3,2)
            plt.cla()
            plt.imshow(flow[:,:,0],vmin=vmin,vmax=vmax)
            plt.title('x_flow')
            plt.pause(.05)
            
            
        n = np.linalg.norm(flow)
        flows.append(flow)
        norms.append(n)

    np.savez(fname[:-4]+'_metrics',flows = flows, norms = norms, correlations = correlations,smoothness=smoothness,
             tmpl = tmpl, smoothness_corr = smoothness_corr, img_corr = img_corr)
    return tmpl, correlations, flows, norms, smoothness


def tile_and_correct_wrapper(params):
    """in parallel"""
    #todo todocumentd
    img_name,  out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie,max_deviation_rigid,upsample_factor_grid, newoverlaps, newstrides, \
        shifts_opencv,nonneg_movie, gSig_filt, is_fiji = params

    imgs = Movie.load(img_name)

    mc = np.zeros(imgs.shape, dtype=np.float32)
    shift_info = []
    for count, img in tqdm(enumerate(imgs)):
        mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       strides=newstrides, upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt)
        shift_info.append([total_shift, start_step, xy_grid])
        
    if out_fname is not None:           
        outv = np.memmap(out_fname, mode='r+', dtype=np.float32, shape=shape_mov, order='F')
        bias = np.float32(add_to_movie) if nonneg_movie else 0
        outv[:, idxs] = np.reshape(mc.astype(np.float32), (len(imgs), -1), order='F').T + bias

    return shift_info, idxs, np.nanmean(mc, axis=0)

