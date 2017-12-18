import numpy as np
import h5py


def read_hdf(file_name, var_name='mov', subindices=None):
    with h5py.File(file_name, "r") as f:
        input_arr = f[var_name]
        input_arr = input_arr[subindices] if subindices is not None else input_arr
        return input_arr


def write_hdf(movie, file_name, var_name='mov'):
    """Save the to an HDF5 (.h5, .hdf, .hdf5) file."""
    with h5py.File(file_name, "w") as f:
        f.create_dataset(var_name, data=np.asarray(movie))

