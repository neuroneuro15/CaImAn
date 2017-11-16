# -*- coding: utf-8 -*-
"""
Class representing a time series.

    Example of usage

    Parameters:
    ----------

    input_arr: np.ndarray

    start_time: time beginning Movie

    fr: frame rate

    meta_data: dictionary including any custom meta data


author: Andrea Giovannucci
"""
from __future__ import print_function

import os
import warnings
import numpy as np




class Timeseries(np.ndarray):
    """
    Class representing a time series.

    Example of usage

    Parameters:
    ----------
    input_arr: np.ndarray

    fr: frame rate

    start_time: time beginning Movie

    meta_data: dictionary including any custom meta data

    Raise:
    -----
    Exception('You need to specify the frame rate')
    """

    def __new__(cls, input_arr, fr=30, start_time=0, file_name=None, meta_data=None):
        """
            Class representing a time series.

            Example of usage

            Parameters:
            ----------
            input_arr: np.ndarray

            fr: frame rate

            start_time: time beginning Movie

            meta_data: dictionary including any custom meta data

            Raise:
            -----
            Exception('You need to specify the frame rate')
            """
        if fr is None:
            raise Exception('You need to specify the frame rate')

        obj = np.asarray(input_arr).view(cls)
        # add the new attribute to the created instance

        obj.start_time = np.double(start_time)
        obj.fr = np.double(fr)
        if type(file_name) is list:
            obj.file_name = file_name
        else:
            obj.file_name = [file_name]

        if type(meta_data) is list:
            obj.meta_data = meta_data
        else:
            obj.meta_data = [meta_data]

        return obj


    @property
    def time(self):
        return np.linspace(self.start_time,1/self.fr*self.shape[0],self.shape[0])

    def __array_prepare__(self, out_arr, context=None):
        # todo: todocument
        inputs=context[1]
        frRef=None
        startRef=None
        for inp in inputs:
            if type(inp) is Timeseries:
                if frRef is None:
                    frRef=inp.fr
                else:
                    if not (frRef-inp.fr) == 0:
                        raise ValueError('Frame rates of input vectors do not match.'
                                         ' You cannot perform operations on time series with different frame rates.')
                if startRef is None:
                    startRef=inp.start_time
                else:
                    if not (startRef-inp.start_time) == 0:
                        warnings.warn('start_time of input vectors do not match: ignore if this is what desired.'
                                      ,UserWarning)

        # then just call the parent
        return np.ndarray.__array_prepare__(self, out_arr, context)


    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

        self.start_time = getattr(obj, 'start_time', None)
        self.fr = getattr(obj, 'fr', None)
        self.file_name = getattr(obj, 'file_name', None)
        self.meta_data = getattr(obj, 'meta_data', None)

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




def concatenate(*args, **kwargs):
    """
    Concatenate movies

    Parameters:
    -----------
    mov: XMovie object
    """
    #todo: todocument return

    obj = []
    frRef = None
    for arg in args:
        for m in arg:
            if issubclass(type(m), Timeseries):
                if frRef is None:
                    obj = m
                    frRef = obj.fr
                else:
                    obj.__dict__['file_name'].extend(
                            [ls for ls in m.file_name])
                    obj.__dict__['meta_data'].extend(
                            [ls for ls in m.meta_data])
                    if obj.fr != m.fr:
                        raise ValueError('Frame rates of input vectors \
                            do not match. You cannot concatenate movies with \
                            different frame rates.')
    try:                      
        return obj.__class__(np.concatenate(*args, **kwargs), **obj.__dict__)
    except:
        print('no meta information passed')
        return obj.__class__(np.concatenate(*args, **kwargs))
