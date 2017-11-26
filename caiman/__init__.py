from __future__ import absolute_import

from caiman.io.mmapping import load_memmap,save_memmap,save_memmap_each,save_memmap_join
from . import io
from .cluster import start_server,stop_server
from .movie import Movie
from .patches import get_patches_from_image, extract_patch_coordinates, apply_to_patch, function_place_holder
from .summary_images import local_correlations


#from .source_extraction import cnmf
