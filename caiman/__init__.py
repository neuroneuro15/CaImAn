from __future__ import absolute_import

from caiman.io.mmapping import load_memmap, save_memmap, save_memmap_each, save_memmap_join
from . import io
from .behavior import select_roi, extract_motor_components_OF, extract_magnitude_and_angle_from_OF, compute_optical_flow
from .cluster import shell_source, Cluster
from .movie import Movie
from .summary_images import local_correlations
#from .source_extraction import cnmf
