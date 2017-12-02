from scipy import io


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat

    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # checks if entries in dictionary rare mat-objects. If yes todict is called to change them to nested dictionaries
    def _todict(matobj):
        """A recursive function which constructs from matobjects nested dictionaries."""

        dd = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, io.matlab.mio5_params.mat_struct):
                dd[strg] = _todict(elem)
            else:
                dd[strg] = elem
        return dd

    for key in data:
        if isinstance(data[key], io.matlab.mio5_params.mat_struct):
            data[key] = _todict(data[key])

    return data
