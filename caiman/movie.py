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

from __future__ import division, print_function, absolute_import

from past.utils import old_div
import cv2
import os
import pickle
import warnings
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import animation
import matplotlib.pyplot as plt
from sklearn import decomposition, cluster, metrics
from scipy import io, optimize
from skimage import feature
from tqdm import tqdm

from .io import sbxreadskip
from .traces import trace
from .mmapping import load_memmap
from .utils import visualization
from .summary_images import local_correlations
from .motion_correction import motion_correct_online


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

    def crop(self, top=0, bottom=0, left=0, right=0, begin=0, end=0):
        """Returns cropped Movie."""
        t, h, w = self.shape
        return self[begin:(t - end), top:(h - bottom), left:(w - right)]

    def to_2D(self,order='F'):
        [T,d1,d2]=self.shape
        d=d1*d2
        return np.reshape(self,(T,d),order=order)

    def to_3d(self, *args, order='F', **kwargs):
        """Synonym for array.reshape()"""
        return self.reshape(*args, order=order, **kwargs)

    def resize(self, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
        # todo: todocument

        # May need to resize in chunks (call self.resize() on smaller slices of Movie) because of OpenCV bug
        max_els = 2 ** 31 - 1
        if self.size > max_els:
            chunk_size =  max_els // np.prod(self.shape[1:])
            new_m, file_names, meta_datas = [], [], []
            for tmp in self[::chunk_size]:
                new_m.append(tmp.resize(fx=fx, fy=fy, fz=fz, interpolation=interpolation))
                file_names.append(tmp.file_name)
                meta_datas.append(tmp.meta_data)
            return self.__class__(np.concatenate(*new_m, axis=0), file_name=file_names, meta_data=meta_datas)

        t, h, w = self.shape
        mov = self.copy()
        if fx != 1 or fy != 1:
            mov = [cv2.resize(frame, (int(w * fy), int(h * fx)), fx=fx, fy=fy, interpolation=interpolation) for frame in self]
        if fz!=1:
            mov = cv2.resize(mov.reshape((t, h * w)), (int(fz * t)), h * w, fx=fz, fy=1, interpolation=interpolation)

        mov = Movie(mov, **self.__dict__)
        mov.fr = self.fr * fz
        return mov

    def bin_median(self):
        """ Return the median image as an array."""
        warnings.warn("Movie.bin_median() deprecated. Use numpy.median(movie) instead.", DeprecationWarning)
        return np.nanmedian(self, axis=0)

    def motion_correction_online(self,max_shift_w=25,max_shift_h=25,init_frames_template=100,
                                 show_movie=False,bilateral_blur=False,template=None,min_count=1000):
        return motion_correct_online(self,max_shift_w=max_shift_w,max_shift_h=max_shift_h,
                                     init_frames_template=init_frames_template,show_movie=show_movie,
                                     bilateral_blur=bilateral_blur,template=template,min_count=min_count)

    def motion_correct(self, max_shift_w=5, max_shift_h=5, num_frames_template=None, template=None, method='opencv',
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
            submov.apply_shifts(shifts, interpolation=interpolation, package=method)
            template=submov.bin_median()
            del submov
            m=self.copy()
            shifts,xcorrs=m.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method=method)
            m=m.apply_shifts(shifts, interpolation=interpolation, package=method)
            template=(m.bin_median())
            del m
        else:
            template=template-np.percentile(template,8)

        # now use the good template to correct
        shifts,xcorrs=self.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method=method)
        self=self.apply_shifts(shifts, interpolation=interpolation, package=method)

        if remove_blanks:
            max_h,max_w= np.max(shifts,axis=0)
            min_h,min_w= np.min(shifts,axis=0)
            self=self.crop(top=max_h, bottom=-min_h + 1, left=max_w, right=-min_w, begin=0, end=0)


        return self,shifts,xcorrs,template

    def extract_shifts(self, max_shift_w=5, max_shift_h=5, template=None, method='opencv'):
        """
        Performs motion correction using the opencv or scikit-image matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.

        Parameters:
        ----------
        max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting in the width and height direction

        template: if a good template for frame by frame correlation is available it can be passed. If None it is automatically computed

        method: depends on what is installed 'opencv' or 'skimage'. 'skimage' is an order of magnitude slower

        Returns:
        -------
        shifts : tuple, contains shifts in x and y and correlation with template

        xcorrs: cross correlation of the movies with the template
        """

        data = self.astype(np.float32) if self.dtype != np.float32 else self

        # Build/Adjust template image
        template_img = np.median(data, axis=0) if type(template) == type(None) else template
        if np.min(template_img) < 0.:
            raise ValueError("All Pixels in Template Array must be greater or equal to zero.")
        template_img = template_img.astype(np.float32)
        template_img = template_img[max_shift_h:(-max_shift_h + 1), max_shift_w:(-max_shift_w + 1)]

        #% run algorithm, press q to stop it
        if method.lower() == 'opencv':
            match_template = lambda img: cv2.matchTemplate(img, template_img, cv2.TM_CCORR_NORMED)
            get_top_left = lambda img: cv2.minMaxLoc(res)[3]
        elif method.lower() == 'skimage':
            match_template = lambda img: feature.match_template(frame, template_img)
            get_top_left = lambda img: np.unravel_index(np.argmax(res), res.shape)[::-1]
        else:
            raise ValueError("Method must be 'opencv' or 'skimage'.")

        shifts, xcorrs = [], []  # store the amount of shift in each frame
        for frame in tqdm(data):
            res = match_template(frame)

            avg_corr = np.mean(res)
            xcorrs.append([avg_corr])

            shift_y, shift_x = get_top_left(frame)
            if (0 < shift_x < 2 * max_shift_h - 1) & (0 < shift_y < 2 * max_shift_w - 1):
                # if max is internal, check for subpixel shift using gaussian
                # peak registration
                log_xm1_y = np.log(res[shift_x - 1., shift_y])
                log_xp1_y = np.log(res[shift_x + 1., shift_y])
                log_x_ym1 = np.log(res[shift_x, shift_y - 1.])
                log_x_yp1 = np.log(res[shift_x, shift_y + 1.])
                four_log_xy = 4. * np.log(res[shift_x, shift_y])

                sh_x_n = max_shift_h - shift_x - (log_xp1_y - log_xm1_y) / (2. * (log_xm1_y - four_log_xy + log_xp1_y))
                sh_y_n = max_shift_w - shift_y - (log_x_yp1 - log_x_ym1) / (2. * (log_x_ym1 - four_log_xy + log_x_yp1))
            else:
                sh_x_n = max_shift_h - shift_x
                sh_y_n = max_shift_w - shift_y

            shifts.append((sh_x_n, sh_y_n))

        return (shifts, xcorrs)

    def apply_shifts(self, shifts, interpolation='linear', package='opencv'):
        """
        Apply precomputed shifts to a Movie in-place, using subpixels adjustment (cv2.INTER_CUBIC function)

        Parameters:
        ------------
        shifts: array of tuples representing x and y shifts for each frame

        interpolation: 'linear', 'cubic', 'nearest' or 'lanczos4'

        Returns:
        -------
        self

        Raise:
        -----
        Exception('Interpolation method not available')

        Exception('Method not defined')
        """
        if package.lower() == 'opencv':
            interp_enum = getattr(cv2, 'INTER_{}'.format(interpolation))
        elif package.lower() == 'skimage':
            interp_enum = {'nearest': 0, 'linear': 1, 'cubic': 3, 'lanczos4': 4}[interpolation]
            from skimage.transform import warp, AffineTransform
        else:
            raise ValueError("'package' argument must be 'opencv' or 'skimage'.")

        if len(shifts) != self.shape[0]:
            raise ValueError("'shifts' argument must have same length as number of frames in movie.")

        t, h, w = self.shape
        shift_mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        for i, (frame, (shift_y, shift_x)) in tqdm(enumerate(zip(self, shifts))):
            if package.lower() == 'opencv':
                shift_mat[:, 2] = shift_x, shift_y
                self[i] = cv2.warpAffine(frame, shift_mat, (w, h), flags=interp_enum, borderMode=cv2.BORDER_REFLECT)
                self[i] = np.clip(self[i], np.min(frame), np.max(frame))
            elif package.lower() == 'skimage':
                tform = AffineTransform(translation=(-shift_x, -shift_y))
                self[i] = warp(frame, tform, preserve_range=True, order=interp_enum, mode='reflect')

    def debleach(self, model='exponential'):
        """
        Debleach in-place by fitting a model to the median intensity.

        Parameters:
        ----------

        model: 'linear' or 'exponential'
        """
        #todo: todocument
        data = self.astype(np.float32, subok=True) if self.dtype != np.float32 else self

        t, h, w = data.shape
        x = np.arange(t)
        y = np.median(self.reshape(t, -1), axis=1)

        if model.lower() == 'linear':
            fit_data = lambda x, a ,b: a * x + b
            p0 = (float(y[-1] - y[0]) / (float(x[-1] - x[0])), y[0])
            popt, pcov = optimize.curve_fit(linf, x, y, p0=p0)
            y_fit = fit_data(x, *popt)
        elif model.lower() == 'exponential':
            fit_data = lambda x, a, b, c: a * np.exp(-b * x) + c
            p0 = (y[0] - y[-1], 1e-6, y[-1])
            popt, pcov = optimize.curve_fit(expf, x, y, p0=p0)
            y_fit = fit_data(x, *popt)
        else:
            raise ValueError("Model must be set to 'linear' or 'exponential'.")

        self.T[:] -= y_fit - np.median(y)

    def computeDFF(self, secsWindow=5, quantilMin=8, method='delta_f_over_f'):
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
        if not method.lower() in ['only_baseline', 'delta_f_over_f', 'delta_f_over_sqrt_f']:
            raise ValueError("Unrecognized method argument: {}".format(method))

        if np.min(self) <= 0 and method != 'only_baseline':
            raise ValueError("All pixels must be positive")

        # compute running baseline
        window = int(secsWindow * self.fr)
        window = window + 1 if window % 2 else window  # just make it even to simplify algorithm.
        half_window = window // 2

        padded_array = np.pad(self, ((half_window, half_window), (0, 0), (0, 0)), mode='reflect')

        t, w, h = padded_array.shape
        stride = self.nbytes // 8
        rolling_array = as_strided(padded_array, shape=(window, t + half_window, w, h), strides=(stride, stride))
        baseline = np.percentile(rolling_array, quantilMin, axis=0)
        baseline = baseline[(half_window - 1):-(half_window - 1), :, :]

        # Compute signal
        out_array = self - baseline
        if method == 'delta_f_over_f':
            out_array /= baseline
        if method == 'delta_f_over_sqrt_f':
            out_array /= np.sqrt(baseline)

        return self.__class__(out_array, **self.__dict__)

    def NMF(self, n_components=30, init='nndsvd', beta=1, tol=5e-7, sparseness='components', **kwargs):
        """
        See documentation for scikit-learn NMF
        """
        if np.min(self) < 0:
            raise ValueError("All values must be positive")

        T, h, w = self.shape
        Y = np.reshape(self, (T, h * w))
        Y = Y - np.percentile(Y, 1)
        Y = np.clip(Y, 0, np.Inf)

        estimator = decomposition.NMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness, **kwargs)
        time_components = estimator.fit_transform(Y)
        space_components = np.reshape(estimator.components_, (n_components, h, w))

        return space_components, time_components

    def NMF_online(self, n_components=30, iterations=-5, **kwargs):
        """ Method performing online matrix factorization and using the spams

        (http://spams-devel.gforge.inria.fr/doc-python/html/index.html) package from Inria.
        Implements bith the nmf and nnsc methods

        Parameters:
        ----------
        n_components: int

        iterations: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        **kwargs: more arguments to be passed to nmf or nnsc

        Return:
        -------
        time_comps

        space_comps
        """
        import spams

        X = self.reshape((self.shape[0], -1), order='F')
        time_comps, V = spams.nmf(X, return_lasso=True, K=n_components, numThreads=4, iter=iterations, **kwargs)
        space_comps = np.array(np.reshape(mm.todense(), self.shape[1:], order='F' for mm in V)
        return time_comps, space_comps

    def NNSC_online(self, n_components=30, lambda1=100, iterations=-5, model=None, **kwargs):
        """ Method performing online matrix factorization and using the spams

        (http://spams-devel.gforge.inria.fr/doc-python/html/index.html) package from Inria.
        Implements bith the nmf and nnsc methods

        Parameters:
        ----------
        n_components: int

        method: 'nnsc' or 'nmf' (see http://spams-devel.gforge.inria.fr/doc-python/html/index.html)

        lambda1: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        iterations: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        model: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

        **kwargs: more arguments to be passed to nmf or nnsc

        Return:
        -------
        time_comps

        space_comps
        """
        import spams

        X = self.reshape(self.shape[0], -1, order='F')
        time_comps, V = spams.nnsc(X, return_lasso=True, K=n_components, lambda1=lambda1, iter=iterations, model=model, **kwargs)
        space_comps = np.array(np.reshape(mm.todense(), self.shape[1:], order='F' for mm in V)
        return time_comps, space_comps


    def IPCA(self, components=50, batch=1000):
        """
        Iterative Principal Component analysis for SVD approximation (see sklearn.decomposition.incremental_pca)
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
        ipca_f = decomposition.incremental_pca(n_components=components, batch_size=batch)
        frames = self.reshape(-1, self.shape[0])
        ipca_f.fit(frames)

        # construct the reduced version of the Movie vectors using only the principal component projection
        proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frames))

        # get the temporal principal components (pixel time series) and associated singular values
        eigenseries = ipca_f.components_.T

        # the rows of eigenseries are approximately orthogonal,
        # so we can approximately obtain eigenframes by multipling the projected frame matrix by this transpose.
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

        ica = decomposition.FastICA(n_components=componentsICA, fun=ICAfun,**kwargs)
        joint_ics = ica.fit_transform(eigenstuff)

        # extract the independent frames
        num_frames, h, w = np.shape(self);
        frame_size = h * w;
        ind_frames = joint_ics[:frame_size, :]
        ind_frames = np.reshape(ind_frames.T, (componentsICA, h, w))

        return ind_frames

    def IPCA_denoise(self, components = 50, batch = 1000):
        """Returns a denoised version of the Movie only using the first 'components' components"""
        _, _, clean_vectors = self.IPCA(components, batch)
        mov = self.__class__(np.reshape(np.float32(clean_vectors.T), np.shape(self)),**self.__dict__)
        return mov

    def IPCA_io(self, n_components=50, fun='logcosh', max_iter=1000, tol=1e-20):
        """ DO NOT USE STILL UNDER DEVELOPMENT"""
        raise NotImplementedError()

#         pca_comp=n_components;
#         [T,d1,d2]=self.shape
#         M=np.reshape(self,(T,d1*d2))
#         [U,S,V] = scipy.sparse.linalg.svds(M,pca_comp)
#         S=np.diag(S);
# #        whiteningMatrix = np.dot(scipy.linalg.inv(np.sqrt(S)),U.T)
# #        dewhiteningMatrix = np.dot(U,np.sqrt(S))
#         whiteningMatrix = np.dot(scipy.linalg.inv(S),U.T)
#         dewhiteningMatrix = np.dot(U,S)
#         whitesig =  np.dot(whiteningMatrix,M)
#         wsigmask=np.reshape(whitesig.T,(d1,d2,pca_comp));
#         f_ica=sklearn.decomposition.FastICA(whiten=False, fun=fun, max_iter=max_iter, tol=tol)
#         S_ = f_ica.fit_transform(whitesig.T)
#         A_ = f_ica.mixing_
#         A=np.dot(A_,whitesig)
#         mask=np.reshape(A.T,(d1,d2,pca_comp)).transpose([2,0,1])
#
#         return mask

    def local_correlations(self, eight_neighbours=False, swap_dim=True, frames_per_chunk=1500):
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
        return local_correlations(self, eight_neighbours=eight_neighbours, swap_dim=swap_dim)

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
        distanceMatrix = metrics.pairwise.euclidean_distances(coordmat.T)
        distanceMatrix=old_div(distanceMatrix,np.max(distanceMatrix))
        estim = cluster.KMeans(n_clusters=n_clusters,max_iter=max_iter)
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

    def guided_filter_blur_2D(self,guide_filter,radius=5, eps=0):
        """
        performs guided filtering on each frame. See opencv documentation of cv2.ximgproc.guidedFilter
        """
        for idx,fr in tqdm(enumerate(self)):
            self[idx] =  cv2.ximgproc.guidedFilter(guide_filter,fr,radius=radius,eps=eps)

        return self

    def bilateral_blur_2D(self,diameter=5,sigmaColor=10000,sigmaSpace=0):
        """
        performs bilateral filtering on each frame. See opencv documentation of cv2.bilateralFilter
        """
        if type(self[0,0,0]) is not np.float32:
            warnings.warn('Casting the array to float 32')
            self=np.asanyarray(self,dtype=np.float32)

        for idx,fr in tqdm(enumerate(self)):
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

        for idx,fr in tqdm(enumerate(self)):
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

        for idx,fr in tqdm(enumerate(self)):
            self[idx] = cv2.medianBlur(fr,ksize=kernel_size)

        return self

    def local_correlations_movie(self,window=10):
        T,_,_=self.shape
        return Movie(np.concatenate([self[j:j + window, :, :].local_correlations(
            eight_neighbours=True)[np.newaxis, :, :] for j in range(T-window)], axis=0), fr=self.fr)

    def plot_aggregation(self, method='mean', **plot_kwargs):
        """
        Compute and plot projection across time:

        Parameters:
        ------------
        method: String, name of numpy aggregation function to use. (Ex: 'mean','median','std')

        **kwargs: Matplotlib arguments to the 'imshow' plotting function.
        """
        agg_data = getattr(np, method)(self, axis=0)
        plt.imshow(agg_data, **plot_kwargs)

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

    def play_notebook(self, speed=1., gain=1.):
        """Returns matplotlib figure with animation of Movie."""

        fig = plt.figure()
        im = plt.imshow(self[0], interpolation='None', cmap=plt.cm.gray)
        plt.axis('off')

        anim = animation.FuncAnimation(fig, lambda frame: (im.set_data(frame * gain),), frames=self, interval=1, blit=True)
        visualization.display_animation(anim, fps= int(self.fr * speed))
        return fig

    def play_opencv(self, speed=1., gain=1.):
        """Create OpenCV window and begin playing Movie.  Press Q key to quit and close window."""

        gain = float(gain)
        maxmov = np.nanmax(self)
        for frame in self:
            cv2.imshow('frame', frame * gain / maxmov)
            if cv2.waitKey(int(speed / self.fr * 1000)) & 0xFF == ord('q'):
                break

        cv2.waitKey(100)
        cv2.destroyAllWindows()
        for i in range(10):
            cv2.waitKey(100)

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

        input_arr = io.loadmat(file_name)['data']
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
