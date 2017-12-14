from __future__ import absolute_import

from caiman.io.mmapping import load_memmap, save_memmap, save_memmap_each, save_memmap_join
from . import io
from .cluster import shell_source, Cluster
from .movie import Movie
from .patches import get_patches_from_image, extract_patch_coordinates, apply_to_patch
from .summary_images import local_correlations
from .behavior import select_roi, extract_motor_components_OF, extract_magnitude_and_angle_from_OF, compute_optical_flow
from .parallel import parallel_dot_product
from .tile_correction import low_pass_filter, sliding_window, create_weight_matrix_for_blending, apply_shift_iteration, register_translation, tile_and_correct, apply_shift_dft
#from .source_extraction import cnmf
