try:
    import tifffile
except ImportError:
    print('tifffile not found, using skimage.externals')
    from skimage.external import tifffile as tifffile
