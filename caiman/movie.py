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

import cv2
from os import path
import pickle
import warnings
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import animation
import matplotlib.pyplot as plt
from sklearn import decomposition, cluster, metrics
from scipy import io, optimize
from tqdm import tqdm

from .io import sbxreadskip, tifffile, load_memmap, save_memmap, read_avi, write_avi
from .summary_images import local_correlations


class Movie(object):

    def __init__(self, input_arr, fr=30, start_time=0, file_name=None, meta_data=None, **kwargs):
        """
        Class representing a Movie. Has a numpy.ndarray-like interface.

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

        self._values = np.array(input_arr, dtype=np.float32) if not isinstance(input_arr, np.memmap) else input_arr
        if self._values.dtype != np.float32:
            raise TypeError('Array data must be encoded as float32 values.')

        self.start_time = np.double(start_time)
        self.fr = np.double(fr)
        self.file_name = file_name if isinstance(file_name, list) else [file_name]
        self.meta_data = meta_data if isinstance(meta_data, list) else [meta_data]

    @property
    def values(self):
        return self._values.view()

    def __getattr__(self, item):
        return getattr(self.values, item)

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    @property
    def time(self):
        return np.linspace(self.start_time, 1 / self.fr * self.shape[0], self.shape[0])

    def crop(self, top=0, bottom=0, left=0, right=0, begin=0, end=0):
        """Returns cropped Movie."""
        t, h, w = self.shape
        return self[begin:(t - end), top:(h - bottom), left:(w - right)]

    def to_2d(self, order='F'):
        T, d1, d2 = self.shape
        d = d1 * d2
        return np.reshape(self, (T, d), order=order)

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
        space_comps = np.array(np.reshape(mm.todense(), self.shape[1:], order='F') for mm in V)
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
        space_comps = np.array(np.reshape(mm.todense(), self.shape[1:], order='F') for mm in V)
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

    def IPCA_stICA(self, componentsPCA=50, componentsICA=40, batch=1000, mu=1, ICAfun='logcosh', **kwargs):
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
        eigenseries, eigenframes, _ = self.IPCA(componentsPCA, batch)

        # normalize the series
        frame_scale = mu / float(np.max(eigenframes))
        frame_mean = np.mean(eigenframes, axis=0)
        n_eigenframes = frame_scale * (eigenframes - frame_mean)

        series_scale = (1. - mu) / np.max(eigenseries)
        series_mean = np.mean(eigenseries, axis = 0)
        n_eigenseries = series_scale * (eigenseries - series_mean)

        # build new features from the space/time data and compute ICA on them
        eigenstuff = np.concatenate([n_eigenframes, n_eigenseries])

        ica = decomposition.FastICA(n_components=componentsICA, fun=ICAfun, **kwargs)
        joint_ics = ica.fit_transform(eigenstuff)

        # extract the independent frames
        num_frames, h, w = np.shape(self)
        ind_frames = joint_ics[:(h * w), :]
        ind_frames = np.reshape(ind_frames.T, (componentsICA, h, w))

        return ind_frames

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

    def partition_FOV_KMeans(self, tradeoff_weight=.5, n_clusters=4, max_iter=500):
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
        T, h, w = self.shape

        idxA, idxB = np.meshgrid(np.arange(w), np.arange(h))
        coordmat = np.vstack((idxA.flatten(), idxB.flatten()))
        distanceMatrix = metrics.pairwise.euclidean_distances(coordmat.T)
        distanceMatrix /= np.max(distanceMatrix)

        mcoef = np.corrcoef(self.T)
        fit_params = tradeoff_weight * mcoef + (tradeoff_weight - 1) * distanceMatrix
        kk = cluster.KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(fit_params)
        fovs = np.reshape(kk.labels_, (h, w))
        return np.uint8(fovs), mcoef, distanceMatrix

    def extract_movie_from_masks(self, masks):
        """
        Parameters:
        ----------------------
        masks: array, 3D with each 2D slice bein a mask (integer or fractional)

        Outputs:
        ----------------------
        traces: array, 2D of fluorescence traces
        """
        T, h, w = self.shape
        Y = np.reshape(self, (T, h * w))

        masks = np.reshape(masks, (-1, np.prod(np.shape(masks)[1:])))
        masks = masks / np.sum(masks, axis=1, keepdims=True) # obtain average over ROI
        masks = np.dot(masks, np.transpose(Y)).T

        return self.__class__(masks, **self.__dict__)

    def guided_filter_blur_2D(self, guide_filter, radius=5, eps=0):
        """Returns a guided-filtered version of the Movie using OpenCV's ximgproc.guidedFilter()."""
        mov = self.copy()
        for frame in tqdm(mov):
            frame[:] = cv2.ximgproc.guidedFilter(guide_filter, frame, radius=radius, eps=eps)
        return self.__class__(mov, **self.__dict__)

    def bilateral_blur_2D(self,diameter=5, sigmaColor=10000, sigmaSpace=0):
        """Returns a bilaterally-filtered version of the Movie using openCV's bilateralFilter() function."""
        mov = self.astype(np.float32)
        for frame in tqdm(mov):
            frame[:] = cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
        return self.__class__(mov, **self.__dict__)

    def gaussian_blur_2D(self, kernel_size_x=5, kernel_size_y=5, kernel_std_x=1, kernel_std_y=1, borderType=cv2.BORDER_REPLICATE):
        """Returns a gaussian-blurred version of the Movie using openCV's GaussianBlur() function."""
        mov = self.copy()
        for frame in tqdm(mov):
            frame[:] = cv2.GaussianBlur(frame, ksize=(kernel_size_x, kernel_size_y), sigmaX=kernel_std_x,
                                         sigmaY=kernel_std_y, borderType=borderType)
        return self.__class__(mov, **self.__dict__)

    def median_blur_2D(self, kernel_size=3):
        """Returns a meduian-blurred version of the Movie using openCV's medianBlur() function."""
        mov = self.copy()
        for frame in tqdm(mov):
            frame[:] = cv2.medianBlur(frame, ksize=kernel_size)
        return self.__class__(mov, **self.__dict__)

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

    def play_notebook(self, speed=1., gain=1., repeat=False):
        """Returns matplotlib figure with animation of Movie."""
        from IPython.display import HTML

        fig = plt.figure()
        im = plt.imshow(self[0], interpolation='None', cmap=plt.cm.gray)
        plt.axis('off')

        anim = animation.FuncAnimation(fig, lambda frame: (im.set_data(frame * gain),), frames=self, interval=1, blit=True)
        plt.close(anim._fig)
        vis = HTML(anim.to_html_video(interval=float(speed) / self.fr, repeat=repeat))
        return vis

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

        input_arr = tifffile.imread(file_name)

        if subindices is not None:
            if isinstance(subindices, list):
                input_arr = input_arr[subindices[0], subindices[1], subindices[2]]
            else:
                input_arr = input_arr[subindices, :, :]
        input_arr = np.squeeze(input_arr)

        return cls(input_arr, fr=fr, start_time=start_time, file_name=path.split(file_name)[-1],
                     meta_data=meta_data)

    @classmethod
    def from_avi(cls, file_name, fr=30, start_time=0, meta_data=None):
        """Loads Movie from a .avi video file."""
        input_arr = io.read_avi(file_name)
        return cls(input_arr, fr=fr, start_time=start_time, file_name=path.split(file_name)[-1], meta_data=meta_data)

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

        return cls(input_arr, fr=fr, start_time=start_time, file_name=path.split(file_name)[-1], meta_data=meta_data)

    @classmethod
    def from_matlab(cls, file_name, fr=30, start_time=0, meta_data=None, subindices=None):

        input_arr = io.loadmat(file_name)['data']
        input_arr = np.rollaxis(input_arr, 2, -3)
        input_arr = input_arr[subindices] if subindices is not None else input_arr
        return cls(input_arr, fr=fr, start_time=start_time, file_name=path.split(file_name)[-1], meta_data=meta_data)


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
                return cls(input_arr, fr=fr, start_time=start_time, file_name=path.split(file_name)[-1],
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
    def from_sbx(cls, file_name, fr=30, subindices=None):
        input_arr = sbxreadskip(file_name[:-4])
        skip = subindices.step if subindices else None
        n_frames = None if subindices else np.inf
        k = None if subindices else 0
        return cls(input_arr, file_names=file_name, fr=fr, k=k, n_frames=n_frames, skip=skip)

    @classmethod
    def from_sima(cls, file_name, fr=30, subindices=None, frame_step=1000, start_time=None, meta_data=None):
        array = read_sima(file_name, subindices=subindices, frame_step=frame_step)
        return cls(array, fr=fr, start_time=start_time, file_name=file_name, meta_data=meta_data)

    @classmethod
    def from_memmap(cls, file_name, mode='r', in_memory=True, fr=30, start_time=0, meta_data=None):
        """Returns Movie from a file created by Movie.to_memmap()."""
        Yr, shape, T = load_memmap(filename=file_name, mode=mode, in_memory=in_memory)
        return cls(Yr, fr=fr, start_time=start_time, meta_data=meta_data, file_name=file_name)

    @classmethod
    def load(cls, file_name, fr=30, start_time=0, meta_data=None, subindices=None, shape=None, var_name_hdf5='mov',
             in_memory=False, is_behavior=False, frame_step_sima=1000):

        name, extension = path.splitext(file_name)[:2]

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
        raise NotImplementedError("Function currently broken. ")
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
        _, extension = path.splitext(file_name)
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
        if to32:
            np.clip(self, np.percentile(self, 1), np.percentile(self, 99.99999), self)
            minn, maxx = np.min(self), np.max(self)
            data = 65536 * (self - minn) / (maxx - minn)
            data = data.astype(np.int32)  # todo: Fix unused data variable.  What is supposed to happen here?
            tifffile.imsave(file_name, self.astype(np.float32))
        else:
            tifffile.imsave(file_name, self)

    def to_npz(self, file_name):
        """Save the Timeseries in a NumPy .npz array file."""
        np.savez(file_name, input_arr=self, start_time=self.start_time, fr=self.fr, meta_data=self.meta_data,
                 file_name=self.file_name)  # todo: check what the two file_name inputs mean.

    def to_avi(self, file_name):
        """Save the Timeseries in a .avi movie file using OpenCV."""
        write_avi(self, file_name, frame_rate=self.fr)

    def to_matlab(self, file_name):
        """Save the Timeseries to a .mat file."""
        f_name = self.file_name if self.file_name[0] is not None else ''
        io.savemat(file_name, {'input_arr': np.rollaxis(self, axis=0, start=3),
                            'start_time': self.start_time,
                            'fr': self.fr,
                            'meta_data': [] if self.meta_data[0] is None else self.meta_data,
                            'file_name': f_name
                            }
                )

    def to_hdf(self, file_name):
        """Save the Timeseries to an HDF5 (.h5, .hdf, .hdf5) file."""
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

    def to_memmap(self, base_filename, order='F', n_chunks=1):
        """Saves efficiently a caiman Movie file into a Numpy memory mappable file by calling caiman.io.save_memmap()

        Parameters:
        ----------
            base_filename: string
                filename to save memory-mapped array to.  (Note: final filename will have shape info in it, and will be returned)

            order: string
                whether to save the file in 'C' or 'F' order

        Returns:
        -------
            fname_tot: the final filename of the mapped file, the format is such that
                the name will contain the frame dimensions and the number of f
        """
        fname_tot = save_memmap(self._values, order=order, n_chunks=n_chunks)
        return fname_tot