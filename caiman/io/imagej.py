import os
import tempfile
import shutil
import zipfile
import numpy as np
from skimage.draw import polygon


def nf_read_roi(fileobj):
    '''
    points = read_roi(fileobj)
    Read ImageJ's ROI format

    Addapted from https://gist.github.com/luispedro/3437255
    '''
    # This is based on:
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html


    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256

    pos = [4]

    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != 'Iout':
        #        raise IOError('Magic number not found')
        print('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used

    roi_type = get8()
    # Discard second Byte:
    get8()

    #    if not (0 <= roi_type < 11):
    #        print(('roireader: ROI type %s not supported' % roi_type))
    #
    #    if roi_type != 7:
    #
    #        print(('roireader: ROI type %s not supported (!= 7)' % roi_type))

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat()
    y1 = getfloat()
    x2 = getfloat()
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:, 1] = [getc() for i in range(n_coordinates)]
    points[:, 0] = [getc() for i in range(n_coordinates)]
    points[:, 1] += left
    points[:, 0] += top
    points -= 1

    return points


# %%
def nf_read_roi_zip(fname, dims, return_names=False):
    # todo todocument

    with zipfile.ZipFile(fname) as zf:
        names = zf.namelist()
        coords = [nf_read_roi(zf.open(n))
                  for n in names]

    def tomask(coords):
        mask = np.zeros(dims)
        coords = np.array(coords)
        rr, cc = polygon(coords[:, 0] + 1, coords[:, 1] + 1)
        mask[rr, cc] = 1

        return mask

    masks = np.array([tomask(s - 1) for s in coords])
    if return_names:
        return masks, names
    else:
        return masks


# %%
def nf_merge_roi_zip(fnames, idx_to_keep, new_fold):
    """
    Create a zip file containing ROIs for ImageJ by combining elements from a list of ROI zip files

    Parameters:
    -----------
        fnames: str
            list of zip files containing ImageJ rois

        idx_to_keep:   list of lists
            for each zip file index of elements to keep

        new_fold: str
            name of the output zip file (without .zip extension)

    """
    folders_rois = []
    files_to_keep = []
    # unzip the files and keep only the ones that are requested
    for fn, idx in zip(fnames, idx_to_keep):
        dirpath = tempfile.mkdtemp()
        folders_rois.append(dirpath)
        with zipfile.ZipFile(fn) as zf:
            name_rois = zf.namelist()
            print(len(name_rois))
        zip_ref = zipfile.ZipFile(fn, 'r')
        zip_ref.extractall(dirpath)
        files_to_keep.append([os.path.join(dirpath, ff) for ff in np.array(name_rois)[idx]])
        zip_ref.close()

    os.makedirs(new_fold)
    for fls in files_to_keep:
        for fl in fls:
            shutil.move(fl, new_fold)
    shutil.make_archive(new_fold, 'zip', new_fold)
    shutil.rmtree(new_fold)

