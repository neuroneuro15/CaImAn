import numpy as np

def read_sima(file_name, subindices=None, frame_step=1000):
    import sima
    dset = sima.ImagingDataset.load(file_name)
    if subindices is None:
        dset_shape = dset.sequences[0].shape
        movie = np.empty((dset_shape[0], dset_shape[2], dset_shape[3]), dtype=np.float32)
        for nframe in range(0, dset.sequences[0].shape[0], frame_step):
            movie[nframe:nframe + frame_step] = np.array(dset.sequences[0][nframe:nframe + frame_step, 0, :, :, 0],
                                                             dtype=np.float32).squeeze()
    else:
        movie = np.array(dset.sequences[0])[subindices, :, :, :, :].squeeze()

    return movie


