# -*- coding: utf-8 -*-
""" Suite of functions that are helpfull work with and manage the Movie

Contains the Movie class.

See Also:
------------

@url
.. image::
@author andrea giovannucci , deep-introspection
"""
# \package caiman/dource_ectraction/cnmf
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Tue Jun 30 20:56:07 2015 , Updated on Fri Aug 19 17:30:11 2016

from __future__ import division, print_function

from past.utils import old_div
import cv2
import os
import sys
import scipy.ndimage
import scipy
import sklearn
import warnings
import numpy as np
import scipy as sp
from sklearn.decomposition import NMF, incremental_pca , FastICA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import h5py
import pickle
from scipy.io import loadmat
from matplotlib import animation
import matplotlib.pyplot as plt
from tqdm import tqdm
from caiman.base import timeseries

from skimage.transform import warp, AffineTransform
from skimage.feature import match_template

from caiman.base.io_sbx import sbxreadskip
from .traces import trace

from caiman.mmapping import load_memmap
from caiman.utils import visualization
from caiman import summary_images as si
from caiman.motion_correction import apply_shift_online,motion_correct_online

class Movie(np.ndarray):
    """
    Class representing a Movie. This class subclasses Timeseries,
    that in turn subclasses ndarray

    Movie(input_arr, fr=None,start_time=0,file_name=None, meta_data=None)

    Example of usage:
    ----------
    input_arr = 3d ndarray
    fr=33; # 33 Hz
    start_time=0
    m=Movie(input_arr, start_time=0,fr=33);

    Parameters:
    ----------

    input_arr:  np.ndarray, 3D, (time,height,width)

    fr: frame rate

    start_time: time beginning Movie, if None it is assumed 0

    meta_data: dictionary including any custom meta data

    file_name: name associated with the file (e.g. path to the original file)

    """

    def __new__(cls, input_arr, fr=30, start_time=0, file_name=None, meta_data=None, **kwargs):
 #todo: todocument
        if (type(input_arr) is np.ndarray) or \
           (type(input_arr) is h5py._hl.dataset.Dataset) or\
           ('mmap' in str(type(input_arr))) or\
           ('tifffile' in str(type(input_arr))):
            obj = np.asarray(input_arr).view(cls)
            obj.start_time = np.double(start_time)
            obj.fr = np.double(fr)
            obj.file_name = file_name if isinstance(file_name, list) else [file_name]
            obj.meta_data = meta_data if isinstance(meta_data, list) else [meta_data]
            return obj
        else:
            raise Exception('Input must be an ndarray, use load instead!')

    def __array_prepare__(self, out_arr, context):
        """Checks that frame rate value given makes sense."""
        if len(set(input.fr for input in context[1] if isinstance(input, self.__class__))) > 1:
            raise ValueError("Frame rates of input vectors must all match each other.")
        if len(set(input.start_time for input in context[1] if isinstance(input, self.__class__))) > 1:
            warnings.warn('start_time of input vectors do not match each other.', UserWarning)

        super(self.__class__, self).__array_prepare__(out_arr, context)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

        self.start_time = getattr(obj, 'start_time', None)
        self.fr = getattr(obj, 'fr', None)
        self.file_name = getattr(obj, 'file_name', None)
        self.meta_data = getattr(obj, 'meta_data', None)

    @property
    def time(self):
        return np.linspace(self.start_time, 1 / self.fr * self.shape[0], self.shape[0])

    def motion_correction_online(self,max_shift_w=25,max_shift_h=25,init_frames_template=100,
                                 show_movie=False,bilateral_blur=False,template=None,min_count=1000):
        return motion_correct_online(self,max_shift_w=max_shift_w,max_shift_h=max_shift_h,
                                     init_frames_template=init_frames_template,show_movie=show_movie,
                                     bilateral_blur=bilateral_blur,template=template,min_count=min_count)

    def apply_shifts_online(self,xy_shifts,save_base_name=None):
        # todo: todocument

        if save_base_name is None:
            return Movie(apply_shift_online(self, xy_shifts, save_base_name=save_base_name), fr=self.fr)
        else:
            return apply_shift_online(self,xy_shifts,save_base_name=save_base_name)

    def calc_min(self):
        # todo: todocument

        tmp = []
        bins = np.linspace(0, self.shape[0], 10).round(0)
        for i in range(9):
            tmp.append(np.nanmin(self[np.int(bins[i]):np.int(bins[i+1]), :, :]).tolist() + 1)
        minval = np.ndarray(1)
        minval[0] = np.nanmin(tmp)
        return Movie(input_arr = minval)
            
    def motion_correct(self,
                       max_shift_w=5,
                       max_shift_h=5,
                       num_frames_template=None,
                       template=None,
                       method='opencv',
                       remove_blanks=False,interpolation='cubic'):

        """
        Extract shifts and motion corrected Movie automatically,

        for more control consider the functions extract_shifts and apply_shifts
        Disclaimer, it might change the object itself.

        Parameters:
        ----------
        max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting
                                 in the width and height direction

        template: if a good template for frame by frame correlation exists
                  it can be passed. If None it is automatically computed

        method: depends on what is installed 'opencv' or 'skimage'. 'skimage'
                is an order of magnitude slower

        num_frames_template: if only a subset of the movies needs to be loaded
                             for efficiency/speed reasons


        Returns:
        -------
        self: motion corected Movie, it might change the object itself

        shifts : tuple, contains x & y shifts and correlation with template

        xcorrs: cross correlation of the movies with the template

        template= the computed template
        """

        if template is None:  # if template is not provided it is created
            if num_frames_template is None:
                num_frames_template = old_div(10e7,(self.shape[1]*self.shape[2]))
                
            frames_to_skip = int(np.maximum(1, old_div(self.shape[0],num_frames_template)))

            # sometimes it is convenient to only consider a subset of the
            # Movie when computing the median
            submov = self[::frames_to_skip, :].copy()
            templ = submov.bin_median() # create template with portion of Movie
            shifts,xcorrs=submov.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=templ, method=method)
            submov.apply_shifts(shifts,interpolation=interpolation,method=method)
            template=submov.bin_median()
            del submov
            m=self.copy()
            shifts,xcorrs=m.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method=method)
            m=m.apply_shifts(shifts,interpolation=interpolation,method=method)
            template=(m.bin_median())
            del m
        else:
            template=template-np.percentile(template,8)

        # now use the good template to correct
        shifts,xcorrs=self.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method=method)
        self=self.apply_shifts(shifts,interpolation=interpolation,method=method)

        if remove_blanks:
            max_h,max_w= np.max(shifts,axis=0)
            min_h,min_w= np.min(shifts,axis=0)
            self=self.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)


        return self,shifts,xcorrs,template


    def bin_median(self,window=10):
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

        """
        T,d1,d2=np.shape(self)
        num_windows=np.int(old_div(T,window))
        num_frames=num_windows*window
        return np.median(np.mean(np.reshape(self[:num_frames],(window,num_windows,d1,d2)),axis=0),axis=0)


    def extract_shifts(self, max_shift_w=5,max_shift_h=5, template=None, method='opencv'):
        """
        Performs motion corretion using the opencv matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.

        Parameters:
        ----------
        max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting in the width and height direction

        template: if a good template for frame by frame correlation is available it can be passed. If None it is automatically computed

        method: depends on what is installed 'opencv' or 'skimage'. 'skimage' is an order of magnitude slower

        Returns:
        -------
        shifts : tuple, contains shifts in x and y and correlation with template

        xcorrs: cross correlation of the movies with the template

        Raise:
        ------
        Exception('Unknown motion correction method!')

        """
        min_val=np.percentile(self, 1)
        if min_val < - 0.1:
            print(min_val)
            warnings.warn('** Pixels averages are too negative. Removing 1 percentile. **')
            self=self-min_val 
        else:
            min_val=0                          

        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float 32')
            self = np.asanyarray(self, dtype=np.float32)

        n_frames_, h_i, w_i = self.shape

        ms_w = max_shift_w
        ms_h = max_shift_h

        if template is None:
            template = np.median(self, axis=0)
        else:
            if np.percentile(template, 8) < - 0.1:
                warnings.warn('Pixels averages are too negative for template. Removing 1 percentile.')
                template=template-np.percentile(template,1)

        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)

        #% run algorithm, press q to stop it
        shifts=[]   # store the amount of shift in each frame
        xcorrs=[]

        for i,frame in enumerate(self):
            if i%100==99:
                print(("Frame %i"%(i+1)))
            if method == 'opencv':
                res = cv2.matchTemplate(frame,template,cv2.TM_CCORR_NORMED)
                top_left = cv2.minMaxLoc(res)[3]
            elif method == 'skimage':
                res = match_template(frame,template)
                top_left = np.unravel_index(np.argmax(res),res.shape)
                top_left=top_left[::-1]
            else:
                raise Exception('Unknown motion correction method!')
            avg_corr=np.mean(res)
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

            shifts.append([sh_x_n,sh_y_n])
            xcorrs.append([avg_corr])

        self=self+min_val

        return (shifts,xcorrs)

    def apply_shifts(self, shifts,interpolation='linear',method='opencv',remove_blanks=False):
        """
        Apply precomputed shifts to a Movie, using subpixels adjustment (cv2.INTER_CUBIC function)

        Parameters:
        ------------
        shifts: array of tuples representing x and y shifts for each frame

        interpolation: 'linear', 'cubic', 'nearest' or cvs.INTER_XXX

        Returns:
        -------
        self

        Raise:
        -----
        Exception('Interpolation method not available')

        Exception('Method not defined')
        """
        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float 32')
            self = np.asanyarray(self, dtype=np.float32)

        if interpolation == 'cubic':
            if method == 'opencv':
                interpolation=cv2.INTER_CUBIC
            else:
                interpolation=3
            print('cubic interpolation')

        elif interpolation == 'nearest':
            if method == 'opencv':
                interpolation=cv2.INTER_NEAREST
            else:
                interpolation=0
            print('nearest interpolation')

        elif interpolation == 'linear':
            if method=='opencv':
                interpolation=cv2.INTER_LINEAR
            else:
                interpolation=1
            print('linear interpolation')
        elif interpolation == 'area':
            if method=='opencv':
                interpolation=cv2.INTER_AREA
            else:
                raise Exception('Method not defined')
            print('area interpolation')
        elif interpolation == 'lanczos4':
            if method=='opencv':
                interpolation=cv2.INTER_LANCZOS4
            else:
                interpolation=4
            print('lanczos/biquartic interpolation')
        else:
            raise Exception('Interpolation method not available')


        t,h,w=self.shape
        for i,frame in enumerate(self):
            if i%100==99:
                print(("Frame %i"%(i+1)));

            sh_x_n, sh_y_n = shifts[i]
            
            if method == 'opencv':
                M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
                min_,max_ = np.min(frame),np.max(frame)
                self[i] = np.clip(cv2.warpAffine(frame,M,(w,h),flags=interpolation,borderMode = cv2.BORDER_REFLECT),min_,max_)

            elif method == 'skimage':

                tform = AffineTransform(translation=(-sh_y_n,-sh_x_n))
                self[i] = warp(frame, tform,preserve_range=True,order=interpolation, borderMode = cv2.BORDER_REFLECT)

            else:
                raise Exception('Unknown shift  application method')

        if remove_blanks:
            max_h,max_w= np.max(shifts,axis=0)
            min_h,min_w= np.min(shifts,axis=0)
            self=self.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)

        return self

    def debleach(self):
        """ Debleach by fiting a model to the median intensity.
        """
    #todo: todocument
        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float 32')
            self = np.asanyarray(self, dtype=np.float32)

        t, h, w = self.shape
        x = np.arange(t)
        y = np.median(self.reshape(t, -1), axis=1)

        def expf(x, a, b, c):
            return a*np.exp(-b*x)+c

        def linf(x, a, b):
            return a*x+b

        try:
            p0 = (y[0]-y[-1], 1e-6, y[-1])
            popt, pcov = sp.optimize.curve_fit(expf, x, y, p0=p0)
            y_fit = expf(x, *popt)
        except:
            p0 = (old_div(float(y[-1]-y[0]),float(x[-1]-x[0])), y[0])
            popt, pcov = sp.optimize.curve_fit(linf, x, y, p0=p0)
            y_fit = linf(x, *popt)

        norm = y_fit - np.median(y[:])
        for frame in range(t):
            self[frame, :, :] = self[frame, :, :] - norm[frame]

        return self

    def crop(self,crop_top=0,crop_bottom=0,crop_left=0,crop_right=0,crop_begin=0,crop_end=0):
        """ Crop Movie
        """
        t,h,w=self.shape
        return self[crop_begin:t-crop_end,crop_top:h-crop_bottom,crop_left:w-crop_right]

    def computeDFF(self,secsWindow=5,quantilMin=8,method='only_baseline',order='F'):
        """
        compute the DFF of the Movie or remove baseline

        In order to compute the baseline frames are binned according to the window length parameter
        and then the intermediate values are interpolated.

        Parameters:
        ----------
        secsWindow: length of the windows used to compute the quantile

        quantilMin : value of the quantile

        method='only_baseline','delta_f_over_f','delta_f_over_sqrt_f'

        Returns:
        -----------
        self: DF or DF/F or DF/sqrt(F) movies

        movBL=baseline Movie

        Raise:
        -----
        Exception('Unknown method')
        """

        print("computing minimum ..."); sys.stdout.flush()
        if np.min(self)<=0 and method != 'only_baseline':
            raise ValueError("All pixels must be positive")
        
        numFrames,linePerFrame,pixPerLine=np.shape(self)
        downsampfact=int(secsWindow*self.fr);
        print(downsampfact)
        elm_missing=int(np.ceil(numFrames*1.0/downsampfact)*downsampfact-numFrames)
        padbefore=int(np.floor(old_div(elm_missing,2.0)))
        padafter=int(np.ceil(old_div(elm_missing,2.0)))

        print(('Inizial Size Image:' + np.str(np.shape(self)))); sys.stdout.flush()
        mov_out=Movie(np.pad(self.astype(np.float32), ((padbefore, padafter), (0, 0), (0, 0)), mode='reflect'), **self.__dict__)
        #mov_out[:padbefore] = mov_out[padbefore+1]
        #mov_out[-padafter:] = mov_out[-padafter-1]
        numFramesNew,linePerFrame,pixPerLine=np.shape(mov_out)

        #% compute baseline quickly
        print("binning data ..."); sys.stdout.flush()
        movBL=np.reshape(mov_out.copy(),(downsampfact,int(old_div(numFramesNew,downsampfact)),linePerFrame,pixPerLine),order=order);
        movBL=np.percentile(movBL,quantilMin,axis=0);
        print("interpolating data ..."); sys.stdout.flush()
        print((movBL.shape))
        movBL=scipy.ndimage.zoom(np.array(movBL,dtype=np.float32),[downsampfact ,1, 1],order=1, mode='constant', cval=0.0, prefilter=False)
#        movBL = Movie(movBL).resize(1,1,downsampfact, interpolation = 4)
        


        #% compute DF/F
        if method == 'delta_f_over_sqrt_f':
            mov_out = old_div((mov_out-movBL),np.sqrt(movBL))
        elif method == 'delta_f_over_f':
            mov_out = old_div((mov_out-movBL),movBL)
        elif method  =='only_baseline':
            mov_out = (mov_out-movBL)
        else:
            raise Exception('Unknown method')

        mov_out=mov_out[padbefore:len(movBL)-padafter,:,:];
        print(('Final Size Movie:' +  np.str(self.shape)))
        return mov_out, Movie(movBL, fr=self.fr, start_time=self.start_time, meta_data=self.meta_data, file_name=self.file_name)

    def computeDFF_trace(self,window_sec=5,minQuantile=20):
        """
        compute the DFF of the Movie

        In order to compute the baseline frames are binned according to the window length parameter
        and then the intermediate values are interpolated.

        Parameters:
        ----------
        secsWindow: length of the windows used to compute the quantile

        quantilMin : value of the quantile

        Raise:
        -----
        ValueError("All traces must be positive")

        ValueError("The window must be shorter than the total length")
        """
        if np.min(self)<=0:
            raise ValueError("All traces must be positive")

        T,num_neurons=self.shape
        window=int(window_sec*self.fr)
        print(window)
        if window >= T:
            raise ValueError("The window must be shorter than the total length")

        tracesDFF=[]
        for tr in self.T:
            print((tr.shape))
            traceBL=[np.percentile(tr[i:i+window],minQuantile) for i in range(1,len(tr)-window)]
            missing=np.percentile(tr[-window:],minQuantile);
            missing=np.repeat(missing,window+1)
            traceBL=np.concatenate((traceBL,missing))
            tracesDFF.append(old_div((tr-traceBL),traceBL))

        return self.__class__(np.asarray(tracesDFF).T,**self.__dict__)


    def NonnegativeMatrixFactorization(self,n_components=30, init='nndsvd', beta=1,tol=5e-7, sparseness='components',**kwargs):
        """
        See documentation for scikit-learn NMF
        """
        if np.min(self)<0:
            raise ValueError("All values must be positive")

        T,h,w=self.shape
        Y=np.reshape(self,(T,h*w))
        Y=Y-np.percentile(Y,1)
        Y=np.clip(Y,0,np.Inf)
        estimator=NMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness,**kwargs)
        time_components=estimator.fit_transform(Y)
        components_ = estimator.components_
        space_components=np.reshape(components_,(n_components,h,w))

        return space_components,time_components

    def online_NMF(self,n_components=30,method='nnsc',lambda1=100,iterations=-5,model=None,**kwargs):
        """ Method performing online matrix factorization and using the spams

        (http://spams-devel.gforge.inria.fr/doc-python/html/index.html) package from Inria.
        Implements bith the nmf and nnsc methods

        Parameters:
        ----------
        n_components: int

        method: 'nnsc' or 'nmf' (see http://spams-devel.gforge.inria.fr/doc-python/html/index.html)

        lambda1: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        iterations: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        batchsize: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        model: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        **kwargs: more arguments to be passed to nmf or nnsc

        Return:
        -------
        time_comps

        space_comps
        """
        try:
            import spams
        except:
            print("You need to install the SPAMS package")
            raise

        T,d1,d2=np.shape(self)
        d=d1*d2
        X=np.asfortranarray(np.reshape(self,[T,d],order='F'))

        if method == 'nmf':
            (time_comps,V) = spams.nmf(X,return_lasso= True ,K = n_components,numThreads=4,iter = iterations,**kwargs)

        elif method == 'nnsc':
            (time_comps,V) = spams.nnsc(X,return_lasso=True,K=n_components, lambda1 = lambda1,iter = iterations, model = model, **kwargs)
        else:
            raise Exception('Method unknown')

        space_comps=[]

        for idx,mm in enumerate(V):
            space_comps.append(np.reshape(mm.todense(),(d1,d2),order='F'))

        return time_comps,np.array(space_comps)


    def IPCA(self, components = 50, batch =1000):
        """
        Iterative Principal Component analysis, see sklearn.decomposition.incremental_pca
        Parameters:
        ------------
        components (default 50) = number of independent components to return

        batch (default 1000)  = number of pixels to load into memory simultaneously in IPCA. More requires more memory but leads to better fit

        Returns:
        -------
        eigenseries: principal components (pixel time series) and associated singular values

        eigenframes: eigenframes are obtained by multiplying the projected frame matrix by the projected Movie (whitened frames?)

        proj_frame_vectors:the reduced version of the Movie vectors using only the principal component projection
        """
        # vectorize the images
        num_frames, h, w = np.shape(self)
        frame_size = h * w
        frame_samples = np.reshape(self, (num_frames, frame_size)).T

        # run IPCA to approxiate the SVD
        ipca_f = incremental_pca(n_components=components, batch_size=batch)
        ipca_f.fit(frame_samples)

        # construct the reduced version of the Movie vectors using only the
        # principal component projection

        proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frame_samples))

        # get the temporal principal components (pixel time series) and
        # associated singular values

        eigenseries = ipca_f.components_.T

        # the rows of eigenseries are approximately orthogonal
        # so we can approximately obtain eigenframes by multiplying the
        # projected frame matrix by this transpose on the right

        eigenframes = np.dot(proj_frame_vectors, eigenseries)

        return eigenseries, eigenframes, proj_frame_vectors

    def IPCA_stICA(self, componentsPCA=50,componentsICA = 40, batch = 1000, mu = 1, ICAfun = 'logcosh', **kwargs):
        """
        Compute PCA + ICA a la Mukamel 2009.



        Parameters:
        -----------
        components (default 50) = number of independent components to return

        batch (default 1000) = number of pixels to load into memory simultaneously in IPCA. More requires more memory but leads to better fit

        mu (default 0.05) = parameter in range [0,1] for spatiotemporal ICA, higher mu puts more weight on spatial information

        ICAFun (default = 'logcosh') = cdf to use for ICA entropy maximization

        Plus all parameters from sklearn.decomposition.FastICA

        Returns:
        --------
        ind_frames [components, height, width] = array of independent component "eigenframes"
        """
        eigenseries, eigenframes,_proj = self.IPCA(componentsPCA, batch)
        # normalize the series

        frame_scale = old_div(mu, np.max(eigenframes))
        frame_mean = np.mean(eigenframes, axis = 0)
        n_eigenframes = frame_scale * (eigenframes - frame_mean)

        series_scale = old_div((1-mu), np.max(eigenframes))
        series_mean = np.mean(eigenseries, axis = 0)
        n_eigenseries = series_scale * (eigenseries - series_mean)

        # build new features from the space/time data
        # and compute ICA on them

        eigenstuff = np.concatenate([n_eigenframes, n_eigenseries])

        ica = FastICA(n_components=componentsICA, fun=ICAfun,**kwargs)
        joint_ics = ica.fit_transform(eigenstuff)

        # extract the independent frames
        num_frames, h, w = np.shape(self);
        frame_size = h * w;
        ind_frames = joint_ics[:frame_size, :]
        ind_frames = np.reshape(ind_frames.T, (componentsICA, h, w))

        return ind_frames

    def IPCA_denoise(self, components = 50, batch = 1000):
        """
        Create a denoise version of the Movie only using the first 'components' components
        """
        _, _, clean_vectors = self.IPCA(components, batch)
        self = self.__class__(np.reshape(np.float32(clean_vectors.T), np.shape(self)),**self.__dict__)
        return self

    def IPCA_io(self, n_components=50, fun='logcosh', max_iter=1000, tol=1e-20):
        """ DO NOT USE STILL UNDER DEVELOPMENT
        """
        pca_comp=n_components;
        [T,d1,d2]=self.shape
        M=np.reshape(self,(T,d1*d2))
        [U,S,V] = scipy.sparse.linalg.svds(M,pca_comp)
        S=np.diag(S);
#        whiteningMatrix = np.dot(scipy.linalg.inv(np.sqrt(S)),U.T)
#        dewhiteningMatrix = np.dot(U,np.sqrt(S))
        whiteningMatrix = np.dot(scipy.linalg.inv(S),U.T)
        dewhiteningMatrix = np.dot(U,S)
        whitesig =  np.dot(whiteningMatrix,M)
        wsigmask=np.reshape(whitesig.T,(d1,d2,pca_comp));
        f_ica=sklearn.decomposition.FastICA(whiten=False, fun=fun, max_iter=max_iter, tol=tol)
        S_ = f_ica.fit_transform(whitesig.T)
        A_ = f_ica.mixing_
        A=np.dot(A_,whitesig)
        mask=np.reshape(A.T,(d1,d2,pca_comp)).transpose([2,0,1])
        
        return mask


    def local_correlations(self,eight_neighbours=False,swap_dim=True, frames_per_chunk = 1500):
        """Computes the correlation image for the input dataset Y

            Parameters:
            -----------

            Y:  np.ndarray (3D or 4D)
                Input Movie data in 3D or 4D format

            eight_neighbours: Boolean
                Use 8 neighbors if true, and 4 if false for 3D data (default = True)
                Use 6 neighbors for 4D data, irrespectively

            swap_dim: Boolean
                True indicates that time is listed in the last axis of Y (matlab format)
                and moves it in the front

            Returns:
            --------

            rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

        """
        T = self.shape[0]
        Cn = np.zeros(self.shape[1:])
        if T<=3000:
            Cn = si.local_correlations(np.array(self), eight_neighbours=eight_neighbours, swap_dim=swap_dim)
        else:
            
            n_chunks = T//frames_per_chunk
            for jj,mv in enumerate(range(n_chunks-1)):
                print('number of chunks:' + str(jj) + ' frames: ' + str([mv*frames_per_chunk,(mv+1)*frames_per_chunk]))
                rho = si.local_correlations(np.array(self[mv*frames_per_chunk:(mv+1)*frames_per_chunk]),
                                            eight_neighbours=eight_neighbours, swap_dim=swap_dim)
                Cn = np.maximum(Cn,rho)    
                plt.imshow(Cn,cmap='gray')
                plt.pause(.1)
            
            print('number of chunks:' + str(n_chunks-1) + ' frames: ' + str([(n_chunks-1)*frames_per_chunk,T]))
            rho = si.local_correlations(np.array(self[(n_chunks-1)*frames_per_chunk:]), eight_neighbours=eight_neighbours,
                                        swap_dim=swap_dim)
            Cn = np.maximum(Cn,rho)    
            plt.imshow(Cn,cmap='gray')
            plt.pause(.1)
            

        return Cn

    def partition_FOV_KMeans(self,tradeoff_weight=.5,fx=.25,fy=.25,n_clusters=4,max_iter=500):
        """
        Partition the FOV in clusters that are grouping pixels close in space and in mutual correlation

        Parameters:
        ------------------------------
        tradeoff_weight:between 0 and 1 will weight the contributions of distance and correlation in the overall metric

        fx,fy: downsampling factor to apply to the Movie

        n_clusters,max_iter: KMeans algorithm parameters

        Outputs:
        -------------------------------
        fovs:array 2D encoding the partitions of the FOV

        mcoef: matric of pairwise correlation coefficients

        distanceMatrix: matrix of picel distances


        Example

        """
        _,h1,w1=self.shape
        self.resize(fx,fy)
        T,h,w=self.shape
        Y=np.reshape(self,(T,h*w))
        mcoef=np.corrcoef(Y.T)

        idxA,idxB =  np.meshgrid(list(range(w)),list(range(h)))
        coordmat=np.vstack((idxA.flatten(),idxB.flatten()))
        distanceMatrix=euclidean_distances(coordmat.T)
        distanceMatrix=old_div(distanceMatrix,np.max(distanceMatrix))
        estim=KMeans(n_clusters=n_clusters,max_iter=max_iter)
        kk=estim.fit(tradeoff_weight*mcoef-(1-tradeoff_weight)*distanceMatrix)
        labs=kk.labels_
        fovs=np.reshape(labs,(h,w))
        fovs=cv2.resize(np.uint8(fovs),(w1,h1),old_div(1.,fx),old_div(1.,fy),interpolation=cv2.INTER_NEAREST)
        return np.uint8(fovs), mcoef, distanceMatrix

    def extract_traces_from_masks(self,masks):
        """
        Parameters:
        ----------------------
        masks: array, 3D with each 2D slice bein a mask (integer or fractional)

        Outputs:
        ----------------------
        traces: array, 2D of fluorescence traces
        """
        T,h,w=self.shape
        Y=np.reshape(self,(T,h*w))
        if masks.ndim == 2:
            masks = masks[None,:,:] 
            
        nA,_,_=masks.shape

        A=np.reshape(masks,(nA,h*w))
        
        pixelsA=np.sum(A,axis=1)
        A=old_div(A,pixelsA[:,None]) # obtain average over ROI
        traces=trace(np.dot(A,np.transpose(Y)).T,**self.__dict__)
        return traces

    def resize(self,fx=1,fy=1,fz=1,interpolation=cv2.INTER_AREA):
        # todo: todocument
        T,d1,d2 =self.shape
        d=d1*d2
        elm=d*T
        max_els=2**31-1
        if elm > max_els:
            chunk_size=old_div((max_els),d)   
            new_m=[]
            print('Resizing in chunks because of opencv bug')
            for chunk in range(0,T,chunk_size):
                print([chunk,np.minimum(chunk+chunk_size,T)])                
                m_tmp=self[chunk:np.minimum(chunk+chunk_size,T)].copy()
                m_tmp=m_tmp.resize(fx=fx,fy=fy,fz=fz,interpolation=interpolation)
                if len(new_m) == 0:
                    new_m=m_tmp
                else:
                    new_m = self.concatenate([new_m,m_tmp],axis=0)

            return new_m
        else:
            if fx!=1 or fy!=1:
                print("reshaping along x and y")
                t,h,w=self.shape
                newshape=(int(w*fy),int(h*fx))
                mov=[]
                print(newshape)
                for frame in self:
                    mov.append(cv2.resize(frame,newshape,fx=fx,fy=fy,interpolation=interpolation))
                self=Movie(np.asarray(mov), **self.__dict__)
            if fz!=1:
                print("reshaping along z")
                t,h,w=self.shape
                self=np.reshape(self,(t,h*w))
                mov=cv2.resize(self,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)
                mov=np.reshape(mov,(np.maximum(1,int(fz*t)),h,w))
                self=Movie(mov, **self.__dict__)
                self.fr=self.fr*fz

        return self
    
    def guided_filter_blur_2D(self,guide_filter,radius=5, eps=0):
        """
        performs guided filtering on each frame. See opencv documentation of cv2.ximgproc.guidedFilter
        """
        for idx,fr in enumerate(self):
            if idx%1000==0:
                print(idx)
            self[idx] =  cv2.ximgproc.guidedFilter(guide_filter,fr,radius=radius,eps=eps)

        return self

    def bilateral_blur_2D(self,diameter=5,sigmaColor=10000,sigmaSpace=0):
        """
        performs bilateral filtering on each frame. See opencv documentation of cv2.bilateralFilter
        """
        if type(self[0,0,0]) is not np.float32:
            warnings.warn('Casting the array to float 32')
            self=np.asanyarray(self,dtype=np.float32)

        for idx,fr in enumerate(self):
            if idx%1000==0:
                print(idx)
            self[idx] =   cv2.bilateralFilter(fr,diameter,sigmaColor,sigmaSpace)

        return self

    def gaussian_blur_2D(self,kernel_size_x=5,kernel_size_y=5,kernel_std_x=1,kernel_std_y=1,borderType=cv2.BORDER_REPLICATE):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Parameters:
        ----------
        kernel_size: double
            see opencv documentation of GaussianBlur
        kernel_std_: double
            see opencv documentation of GaussianBlur
        borderType: int
            see opencv documentation of GaussianBlur

        Returns:
        --------
        self: ndarray
            blurred Movie
        """

        for idx,fr in enumerate(self):
            print(idx)
            self[idx] = cv2.GaussianBlur(fr,ksize=(kernel_size_x,kernel_size_y),sigmaX=kernel_std_x,sigmaY=kernel_std_y,
                                         borderType=borderType)

        return self

    def median_blur_2D(self,kernel_size=3):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Parameters:
        ----------
        kernel_size: double
            see opencv documentation of GaussianBlur

        kernel_std_: double
            see opencv documentation of GaussianBlur

        borderType: int
            see opencv documentation of GaussianBlur

        Returns:
        --------
        self: ndarray
            blurred Movie
        """

        for idx,fr in enumerate(self):
            print(idx)
            self[idx] = cv2.medianBlur(fr,ksize=kernel_size)

        return self

    def resample(self):
        raise NotImplementedError

    def to_2D(self,order='F'):
        [T,d1,d2]=self.shape
        d=d1*d2
        return np.reshape(self,(T,d),order=order)

    def zproject(self,method='mean',cmap=plt.cm.gray,aspect='auto',**kwargs):
        """
        Compute and plot projection across time:

        Parameters:
        ------------
        method: String
            'mean','median','std'

        **kwargs: dict
            arguments to imagesc

        Raise:
        ------
        Exception('Method not implemented')
        """
        # todo: todocument
        if method is 'mean':
            zp=np.mean(self,axis=0)
        elif method is 'median':
            zp=np.median(self,axis=0)
        elif method is 'std':
            zp=np.std(self,axis=0)
        else:
            raise Exception('Method not implemented')
        plt.imshow(zp,cmap=cmap,aspect=aspect,**kwargs)
        return zp

    def local_correlations_movie(self,window=10):
        T,_,_=self.shape
        return Movie(np.concatenate([self[j:j + window, :, :].local_correlations(
            eight_neighbours=True)[np.newaxis, :, :] for j in range(T-window)], axis=0), fr=self.fr)


    def plot_trace(self, stacked=True, subtract_minimum=False, cmap=plt.cm.jet, **kwargs):
        """Plot the data as a trace (Note: experimental method, may not work quite right yet.)

        author: ben deverett

        Parameters:
        ----------
        stacked : bool
            for multiple columns of data, stack instead of overlaying

        subtract_minimum : bool
            subtract minimum from each individual trace

        cmap : matplotlib.LinearSegmentedColormap
            color map for display. Options are found in plt.colormaps(), and are accessed as plt.cm.my_favourite_map

        kwargs : dict
            any arguments accepted by matplotlib.plot

        Returns:
        -------
        The matplotlib axes object corresponding to the data plot
        """
        d = self.copy()
        n = 1 #number of traces
        if len(d.shape)>1:
            n = d.shape[1]

        ax = plt.gca()

        colors = cmap(np.linspace(0, 1, n))
        ax.set_color_cycle(colors)

        if subtract_minimum:
            d -= d.min(axis=0)
        if stacked and n>1:
            d += np.append(0, np.cumsum(d.max(axis=0))[:-1])
        ax.plot(self.time, d, **kwargs)

        # display trace labels along right
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.atleast_1d(d.mean(axis=0)))
        ax2.set_yticklabels([str(i) for i in range(n)], weight='bold')
        [l.set_color(c) for l,c in zip(ax2.get_yticklabels(), colors)]

        plt.gcf().canvas.draw()
        return ax



    def play(self,gain=1,fr=None,magnification=1,offset=0,interpolation=cv2.INTER_LINEAR,
             backend='opencv',do_loop=False, bord_px = None):
        """
        Play the Movie using opencv

        Parameters:
        ----------
        gain: adjust  Movie brightness

        frate : playing speed if different from original (inter frame interval in seconds)

        backend: 'pylab' or 'opencv', the latter much faster

        Raise:
        -----
         Exception('Unknown backend!')
        """
        # todo: todocument
        if backend is 'matplotlib':
            print('*** WARNING *** SPEED MIGHT BE LOW. USE opencv backend if available')

        gain*=1.
        maxmov=np.nanmax(self)

        if backend is 'matplotlib':
            plt.ion()
            fig = plt.figure( 1 )
            ax = fig.add_subplot( 111 )
            ax.set_title("Play Movie")
            im = ax.imshow( (offset+self[0])*gain/maxmov ,cmap=plt.cm.gray,vmin=0,vmax=1,interpolation='none') # Blank starting image
            fig.show()
            im.axes.figure.canvas.draw()
            plt.pause(1)

        if backend is 'notebook':
            # First set up the figure, the axis, and the plot element we want to animate
            fig = plt.figure()
            im = plt.imshow(self[0],interpolation='None',cmap=plt.cm.gray)
            plt.axis('off')
            def animate(i):
                im.set_data(self[i])
                return im,

            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate,
                                           frames=self.shape[0], interval=1, blit=True)

            # call our new function to display the animation
            return visualization.display_animation(anim, fps=fr)


        if fr==None:
            fr=self.fr

        looping=True
        terminated=False

        while looping:

            for iddxx,frame in enumerate(self):
                if bord_px is not None and np.sum(bord_px) > 0:
                    frame = frame[bord_px:-bord_px, bord_px:-bord_px]
                    
                if backend is 'opencv':
                    if magnification != 1:
                        frame = cv2.resize(frame,None,fx=magnification, fy=magnification, interpolation = interpolation)

                    cv2.imshow('frame',(offset+frame)*gain/maxmov)
                    if cv2.waitKey(int(1./fr*1000)) & 0xFF == ord('q'):
                        looping=False
                        terminated=True
                        break

                
                elif backend is 'matplotlib':

                    im.set_data((offset+frame)*gain/maxmov)
                    ax.set_title( str( iddxx ) )
                    plt.axis('off')
                    fig.canvas.draw()
                    plt.pause(1./fr*.5)
                    ev=plt.waitforbuttonpress(1./fr*.5)
                    if ev is not None:
                        plt.close()
                        break

                elif backend is 'notebook':
                    print('Animated via MP4')
                    break

                else:
                    raise Exception('Unknown backend!')

            if terminated:
                break

            if do_loop:
                looping=True


        if backend is 'opencv':
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            for i in range(10):
                cv2.waitKey(100)

    @classmethod
    def concatenate(cls, *series, axis=0):
        """Concatenate multiple TimeSeries objects together."""
        if not all(map(lambda s: isinstance(s, cls), series)):
            raise ValueError("All series must be {} objects in order to concatenate them.".format(cls.__name__))
        if len(set(ts.fr for ts in series)) > 1:
            raise ValueError("Timeseries must all have matching framerates.")

        file_names = [s.file_name for s in series]
        meta_datas = [s.meta_data for s in series]
        return cls(np.concatenate(*series, axis=axis), file_name=file_names, meta_data=meta_datas)

    @classmethod
    def from_tiff(cls, file_name, fr=30, start_time=0, meta_data=None, subindices=None):
        """Loads Movie from a .tiff image file."""
        from skimage.external.tifffile import imread
        input_arr = imread(file_name)

        if subindices is not None:
            if isinstance(subindices, list):
                input_arr = input_arr[subindices[0], subindices[1], subindices[2]]
            else:
                input_arr = input_arr[subindices, :, :]
        input_arr = np.squeeze(input_arr)

        return cls(input_arr, fr=fr, start_time=start_time, file_name=os.path.split(file_name)[-1],
                     meta_data=meta_data)

    @classmethod
    def from_avi(cls, file_name, fr=30, start_time=0, meta_data=None):
        """Loads Movie from a .avi video file."""
        cap = cv2.VideoCapture(file_name)
        use_cv2 = hasattr(cap, 'CAP_PROP_FRAME_COUNT')

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) if use_cv2 else cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) if use_cv2 else cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if use_cv2 else cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        input_arr = np.zeros((length, height, width), dtype=np.uint8)
        for arr in input_arr:
            _, frame = cap.read()
            arr[:] = frame[:, :, 0]

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        return cls(input_arr, fr=fr, start_time=start_time, file_name=os.path.split(file_name)[-1], meta_data=meta_data)

    @classmethod
    def from_npy(cls, file_name, fr=30, start_time=0, in_memory=False, meta_data=None, subindices=None, shape=None):

        input_arr = np.load(file_name) if in_memory else np.load(file_name, mmap_mode='r')
        input_arr = input_arr[subindices] if subindices is not None else input_arr

        if input_arr.ndim == 2:
            if shape is not None:
                d, T = np.shape(input_arr)
                input_arr = np.transpose(np.reshape(input_arr, (shape[0], shape[1], T), order='F'), (2, 0, 1))
            else:
                input_arr = input_arr[np.newaxis, :, :]

        return cls(input_arr, fr=fr, start_time=start_time, file_name=os.path.split(file_name)[-1], meta_data=meta_data)

    @classmethod
    def from_matlab(cls, file_name, fr=30, start_time=0, meta_data=None, subindices=None):

        input_arr = loadmat(file_name)['data']
        input_arr = np.rollaxis(input_arr, 2, -3)
        input_arr = input_arr[subindices] if subindices is not None else input_arr
        return cls(input_arr, fr=fr, start_time=start_time, file_name=os.path.split(file_name)[-1], meta_data=meta_data)


    @classmethod
    def from_npz(cls, file_name):
        with np.load(file_name) as input_arr:
            return cls(input_arr, file_name=file_name)

    @classmethod
    def from_hdf(cls, file_name, meta_data=None, subindices=None, var_name='mov'):
        with h5py.File(file_name, "r") as f:
            input_arr = f[var_name]
            input_arr = input_arr[subindices] if subindices is not None else input_arr

            attrs = dict(f[var_name].attrs)
            if meta_data in attrs:
                attrs['meta_data'] = pickle.loads(attrs['meta_data'])
            return cls(input_arr, **attrs)

    @classmethod
    def from_hdf_at(cls, file_name, fr=30, subindices=None):
        with h5py.File(file_name, "r") as f:
            input_arr = f['quietBlock']
            input_arr = input_arr[subindices] if subindices is not None else input_arr
            return cls(input_arr, fr=fr)

    @classmethod
    def from_h5(cls, file_name, fr=30, subindices=None, is_behavior=False, start_time=None, meta_data=None):

        with h5py.File(file_name, "r") as f:
            if is_behavior:
                kk = f.keys()
                kk.sort(key=lambda x: np.int(x.split('_')[-1]))
                input_arr = [np.array(f[trial]['mov']) for trial in kk]
                input_arr = np.vstack(input_arr)
                return cls(input_arr, fr=fr, start_time=start_time, file_name=os.path.split(file_name)[-1],
                             meta_data=meta_data)
            else:
                if 'imaging' in f.keys():
                    if subindices is None:
                        input_arr = np.array(f['imaging']).squeeze()
                        if input_arr.ndim > 3:
                            input_arr = input_arr[:, 0]
                    else:
                        input_arr = np.array(f['imaging'][subindices]).squeeze()
                        if input_arr.ndim > 3:
                            input_arr = input_arr[:, 0]
                    input_arr = input_arr.astype(np.float32)


            return cls(input_arr, fr=fr, file_name=file_name)  # TODO: Finish this function!

    @classmethod
    def from_mmap(cls, file_name, fr=30, in_memory=False):

        filename = os.path.split(file_name)[-1]
        Yr, dims, T = load_memmap(os.path.join(os.path.split(file_name)[0], filename))
        input_arr = np.reshape(Yr.T, [T] + list(dims), order='F')
        input_arr = np.array(input_arr) if in_memory else input_arr

        return Movie(input_arr, fr=fr)

    @classmethod
    def from_sbx(cls, file_name, fr=30, subindices=None):
        input_arr = sbxreadskip(file_name[:-4])
        skip = subindices.step if subindices else None
        n_frames = None if subindices else np.inf
        k = None if subindices else 0
        return cls(input_arr, file_names=file_name, fr=fr, k=k, n_frames=n_frames, skip=skip)

    @classmethod
    def from_sima(cls, file_name, fr=30, subindices=None, frame_step=1000, start_time=None, meta_data=None):
        import sima
        dset = sima.ImagingDataset.load(file_name)
        if subindices is None:
            dset_shape = dset.sequences[0].shape
            input_arr = np.empty((dset_shape[0], dset_shape[2], dset_shape[3]), dtype=np.float32)
            for nframe in range(0, dset.sequences[0].shape[0], frame_step):
                input_arr[nframe:nframe + frame_step] = np.array(dset.sequences[0][nframe:nframe + frame_step, 0, :, :, 0],
                                                                 dtype=np.float32).squeeze()
        else:
            input_arr = np.array(dset.sequences[0])[subindices, :, :, :, :].squeeze()


        return cls(input_arr, fr=fr, start_time=start_time, file_name=os.path.split(file_name)[-1],
                     meta_data=meta_data)


    @classmethod
    def load(cls, file_name, fr=30, start_time=0, meta_data=None, subindices=None, shape=None, var_name_hdf5='mov',
             in_memory=False, is_behavior=False, frame_step_sima=1000):

        name, extension = os.path.splitext(file_name)[:2]

        if extension == '.tif' or extension == '.tiff':  # load avi file
            return cls.from_tiff(file_name=file_name, fr=fr, start_time=start_time, subindices=subindices, meta_data=meta_data)
        elif extension == '.avi': # load avi file
            return cls.from_avi(file_name=file_name, fr=fr, start_time=start_time, meta_data=meta_data)
        elif extension == '.npy': # load npy file
            return cls.from_npy(file_name=file_name, fr=30, start_time=start_time, meta_data=meta_data, subindices=subindices, shape=shape)
        elif extension == '.mat': # load npy file
            return cls.from_matlab(file_name=file_name, fr=fr, start_time=start_time, meta_data=meta_data, subindices=subindices)
        elif extension == '.npz': # load Movie from saved file
            return cls.from_npz(file_name=file_name)
        elif extension== '.hdf5':
            return cls.from_hdf(file_name=file_name, meta_data=meta_data, subindices=subindices, var_name=var_name_hdf5)
        elif extension== '.h5_at':
            return cls.from_hdf_at(file_name=file_name, fr=fr, subindices=subindices)
        elif extension== '.h5':
            return cls.from_h5(file_name=file_name, fr=fr, subindices=subindices, is_behavior=is_behavior)
        elif extension == '.mmap':
            return cls.from_mmap(file_name=file_name, fr=fr, in_memory=in_memory)
        elif extension == '.sbx':
            return cls.from_sbx(file_name=file_name, fr=fr, subindices=subindices)
        elif extension == '.sima':
            return cls.from_sima(file_name=file_name, fr=fr, subindices=subindices, frame_step=frame_step_sima, start_time=start_time, meta_data=meta_data)
        else:
            raise ValueError('Unknown file type: "{}"'.format(extension))

    @classmethod
    def load_multiple(cls, file_list, fr=30, start_time=0, crop=(0, 0, 0, 0),
                     meta_data=None, subindices=None, channel = None):

        """ load movies from list of file names

        Parameters:
        ----------
        file_list: list
           file names in string format

        the other parameters as in load_movie except

        bottom, top, left, right: int
            to load only portion of the field of view

        Returns:
        --------
        Movie: cm.Movie
            Movie corresponding to the concatenation og the input files

        """
        mov = []
        for f in tqdm(file_list):
            m = cls.load(f, fr=fr, start_time=start_time,
                     meta_data=meta_data, subindices=subindices, in_memory = True)
            m = m[channel].squeeze() if channel is not None else m
            m = m[np.newaxis, :, :] if m.ndim == 2 else m

            tm, height, width = np.shape(m)
            top, bottom, left, right = crop
            m = m[:, top:(height - bottom), left:(width - right)]

            mov.append(m)
        return concatenate(mov, axis=0)

    def save(self, file_name, to32=True, order='F'):
        """
        Save the Timeseries in various formats, depending on the file_name's extenstion.

        parameters:
        ----------
        file_name: str
            name of file. Possible formats are tif, avi, npz and hdf5

        to32: Bool
            whether to transform to 32 bits

        order: 'F' or 'C'
            C or Fortran order

        Raise:
        -----
        raise ValueError('Extension Unknown')

        """
        _, extension = os.path.splitext(file_name)
        if 'tif' in extension:
            self.to_tiff(file_name=file_name, to32=to32)
        elif 'mat' in extension:
            self.to_matlab(file_name=file_name)
        elif 'npz' in extension:
            self.to_npz(file_name=file_name)
        elif 'hdf' in extension or extension == '.h5':
            self.to_hdf(file_name=file_name)
        elif 'avi' in extension:
            self.to_avi(file_name=file_name)
        else:
            raise ValueError('Could Not Save to File: File Extension "{}" Not Supported.'.format(extension))

    def to_tiff(self, file_name, to32=True):
        """Save the Timeseries in a .tiff image file."""
        try:
            from tifffile import imsave
        except ImportError:
            warnings.warn('tifffile package not found, importing skimage instead for saving tiff files.')
            from skimage.external.tifffile import imsave

        if to32:
            np.clip(self, np.percentile(self, 1), np.percentile(self, 99.99999), self)
            minn, maxx = np.min(self), np.max(self)
            data = 65536 * (self - minn) / (maxx - minn)
            data = data.astype(np.int32)  # todo: Fix unused data variable.  What is supposed to happen here?
            imsave(file_name, self.astype(np.float32))
        else:
            imsave(file_name, self)

    def to_npz(self, file_name):
        """Save the Timeseries in a NumPy .npz array file."""
        np.savez(file_name, input_arr=self, start_time=self.start_time, fr=self.fr, meta_data=self.meta_data,
                 file_name=self.file_name)  # todo: check what the two file_name inputs mean.

    def to_avi(self, file_name):
        """Save the Timeseries in a .avi movie file using OpenCV."""
        import cv2
        codec = cv2.FOURCC('I', 'Y', 'U', 'V') if hasattr(cv2, 'FOURCC') else cv2.VideoWriter_fourcc(*'IYUV')
        np.clip(self, np.percentile(self, 1), np.percentile(self, 99), self)
        minn, maxx = np.min(self), np.max(self)
        data = 255 * (self - minn) / (maxx - minn)
        data = data.astype(np.uint8)
        y, x = data[0].shape
        vw = cv2.VideoWriter(file_name, codec, self.fr, (x, y), isColor=True)
        for d in data:
            vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
        vw.release()

    def to_matlab(self, file_name):
        """Save the Timeseries to a .mat file."""
        from scipy.io import savemat
        f_name = self.file_name if self.file_name[0] is not None else ''
        savemat(file_name, {'input_arr': np.rollaxis(self, axis=0, start=3),
                            'start_time': self.start_time,
                            'fr': self.fr,
                            'meta_data': [] if self.meta_data[0] is None else self.meta_data,
                            'file_name': f_name
                            }
                )

    def to_hdf(self, file_name):
        """Save the Timeseries to an HDF5 (.h5, .hdf, .hdf5) file."""
        import pickle
        import h5py
        with h5py.File(file_name, "w") as f:
            dset = f.create_dataset("mov", data=np.asarray(self))
            dset.attrs["fr"] = self.fr
            dset.attrs["start_time"] = self.start_time
            try:
                dset.attrs["file_name"] = [a.encode('utf8') for a in self.file_name]
            except:
                print('No file name saved')
            if self.meta_data[0] is not None:
                print(self.meta_data)
                dset.attrs["meta_data"] = pickle.dumps(self.meta_data)

    def to_3d(self, *args, order='F', **kwargs):
        """Synonym for array.reshape()"""
        return self.reshape(*args, order=order, **kwargs)



def concatenate(*series, axis=0):
    """Concatenate Movies together."""
    return Movie.concatenate(*series, axis=axis)