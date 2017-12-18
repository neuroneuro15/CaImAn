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
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import animation
import matplotlib.pyplot as plt
from sklearn import decomposition, cluster, metrics
from scipy import io, optimize


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