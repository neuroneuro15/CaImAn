from warnings import warn
from tqdm import tqdm
import numpy as np

try:
    import tifffile
except ImportError:
    warn('tifffile standalone not found, so using tifffile from skimage.externals instead.')
    from skimage.external import tifffile as tifffile



def val_parse(v):
    """parse values from si tags into python objects if possible from si parse

     Parameters:
     -----------

     v: si tags

     returns:
     -------

    v: python object

    """
    try:
        return {'true': True, 'false': False, 'NaN': np.nan, 'inf': np.inf, 'Inf': np.inf}[v]
    except KeyError:
        try:
            return eval(v)
        except:
            return v


def si_parse(imd):
    """parse image_description field embedded by scanimage from get iamge description

     Parameters:
     -----------

     imd: image description

     returns:
     -------

    imd: the parsed description

    """
    imd = imd.split('\n')
    imd = (i.split('=') for i in imd if '=' in i)
    imd = ([ii.strip(' \r') for ii in i] for i in imd)
    imd = {i[0]: val_parse(i[1]) for i in imd}
    return imd


def get_image_description_SI(fname):
    """Given a tif file acquired with Scanimage it returns a dictionary containing the information in the image description field

     Parameters:
     -----------

     fname: name of the file

     returns:
     -------

        image_description: information of the image

    Raise:
    -----
        ('tifffile package not found, using skimage.external.tifffile')


    """
    tf = tifffile.TiffFile(fname)
    image_descriptions = [page.tags['image_description'].value for page in tqdm(tf.pages)]
    return image_descriptions

