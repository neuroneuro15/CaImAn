from warnings import warn

try:
    import tifffile
except ImportError:
    warn('tifffile standalone not found, so using tifffile from skimage.externals instead.')
    from skimage.external import tifffile as tifffile
