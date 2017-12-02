import json
import numpy as np


def neurofinder_format_masks(binary_masks):
    """
    Returns a list of ROI 'coordinate' dictionaries from ncomp x height x width, binary mask array, for the neurofinder format.

    For more info on Neurofinder, go to http://neurofinder.codeneuro.org/
    """
    return  [{"coordinates": list(zip(*np.where(m)))} for m in binary_masks]


def neurofinder_masks_to_json(binary_masks, json_filename):
    """
    Saves a tensor of binary mask to the JSON file format for Neurofinder (http://neurofinder.codeneuro.org)

    Parameters:
    -----------
    binary_masks: 3d ndarray (components x dimension 1  x dimension 2)

    json_filename: str

    """
    with open(json_filename, 'w') as f:
        json.dump(neurofinder_format_masks(binary_masks), f)


def neurofinder_load_masks_from_json(json_filename, dims):
    """Load a Neurofinder JSON file to get binary mask array of ROI regions (n x height x width)"""
    with open(json_filename) as f:
        regions = json.load(f)

    masks = np.zeros(dims, dtype=np.bool)
    for roi_idx, coord_dict in enumerate(regions):
        roi_mask = masks[roi_idx, :, :]
        roi_mask[list(zip(*coord_dict['coordinates']))] = 1

    return masks