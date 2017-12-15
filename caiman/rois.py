# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:22:26 2015

@author: agiovann
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import optimize, ndimage, sparse
from skimage import filters
from skimage.morphology import remove_small_objects, opening, remove_small_holes, closing, dilation, watershed
from tqdm import tqdm

from .io import nf_read_roi_zip


def extract_binary_masks_from_structural_channel(img, min_area_size=30, min_hole_size=15, gSig=5, expand_method='closing', selem=np.ones((3,3))):
    """Extract binary masks by using adaptive thresholding on a structural channel
    
    Inputs:
    ------
    img:                  Numpy array
                        Movie of the structural channel (assumed motion corrected)
    
    min_area_size:      int
                        ignore components with smaller size

    min_hole_size:      int
                        fill in holes up to that size (donuts)
                        
    gSig:               int
                        average radius of cell
                        
    expand_method:      string
                        method to expand binary masks (morphological closing or dilation)
                        
    selem:              np.array
                        morphological element with which to expand binary masks
                        
    Output:
    -------
    A:                  sparse column format matrix
                        matrix of binary masks to be used for CNMF seeding
                        
    mR:                 np.array
                        mean image used to detect cell boundaries
    """

    if not expand_method.lower() in ['dilation', 'closing']:
        raise ValueError("expand_method argument must be either 'dilation' or 'closing'.")

    mR = img.mean(axis=0)
    img = cv2.blur(mR,(gSig,gSig))
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
    img = img.astype(np.uint8)
    img = cv2.adaptiveThreshold(img, np.max(img), cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, gSig, 0)
    img = remove_small_holes(img > 0, min_size=min_hole_size)
    img = remove_small_objects(img, min_size=min_area_size)
    areas, n_features = ndimage.label(img)
    
    A = np.zeros((np.prod(img.shape), n_features), dtype=bool)
    for i in range(n_features):
        temp = (areas == i + 1)
        temp = dilation(temp, selem=selem) if expand_method.lower() == 'dilation' else closing(temp, selem=selem)
        A[:, i] = temp.flatten(order='F')
    
    return A, mR


def mask_to_2d(mask):
    #todo todocument
    dims = np.shape(mask)[-2:]
    if mask.ndim > 2:
        mask = mask[:].transpose([1, 2, 0])
    mask = np.reshape(mask, (np.prod(dims), -1,), order='F')
    return sparse.coo_matrix(mask)


def get_distance_from_A(masks_gt, masks_comp, min_dist=10):
    #todo todocument
    dims = np.shape(masks_gt)
    masks_gt_new = np.reshape(masks_gt[:].transpose([1, 2, 0]), (np.prod(dims), -1,), order='F')
    masks_comp_new = np.reshape(masks_comp[:].transpose([1,2,0]), (np.prod(dims), -1,), order='F')

    A_ben = sparse.csc_matrix(masks_gt_new)
    A_cnmf = sparse.csc_matrix(masks_comp_new)

    cm_ben = [ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [ndimage.center_of_mass(mm) for mm in masks_comp]

    return distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf], min_dist)


def nf_match_neurons_in_binary_masks(masks_gt, masks_comp, thresh_cost=.7, min_dist=10, D=None, enclosed_thr=None):
    """
    Match neurons expressed as binary masks. Uses Hungarian matching algorithm

    Parameters:
    -----------

    masks_gt: bool ndarray  components x d1 x d2
        ground truth masks

    masks_comp: bool ndarray  components x d1 x d2
        mask to compare to

    thresh_cost: double 
        max cost accepted 

    min_dist: min distance between cm

    Cn: 
        correlation image or median

    D: list of ndarrays
	list of distances matrices
    
    enclosed_thr: float
        if not None set distance to at most the specified value when ground truth is a subset of inferred        

    Returns:
    --------
    idx_tp_1:
        indeces true pos ground truth mask 

    idx_tp_2:
        indeces true pos comp

    idx_fn_1:
        indeces false neg

    idx_fp_2:
        indeces false pos

    performance:  

    """
    # transpose to have a sparse list of components, then reshaping it to have a 1D matrix read in the Fortran style
    dims = np.shape(masks_gt)[1:]
    A_ben = sparse.csc_matrix(np.reshape(masks_gt.transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))
    A_cnmf = sparse.csc_matrix(np.reshape(masks_comp.transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))

    # have the center of mass of each element of the two masks
    cm_ben = [ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [ndimage.center_of_mass(mm) for mm in masks_comp]

    D = distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf], min_dist, enclosed_thr=enclosed_thr) if D is None else D  # find the distance between each mask
    matches, costs = find_matches(D)
    matches = matches[0]
    costs = costs[0]

    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    idx_tp_gt = matches[0][idx_tp]    # ground truth
    idx_tp_comp = matches[1][idx_tp]   # algorithm - comp
    idx_fn_gt = np.setdiff1d(np.arange(masks_gt.shape[0]), matches[0][idx_tp])
    idx_fp_comp =  np.setdiff1d(np.arange(masks_comp.shape[0]), matches[1][idx_tp])

    # compute precision and recall
    TP = float(np.sum(np.array(costs) < thresh_cost))
    FN = np.shape(masks_gt)[0] - TP
    FP = np.shape(masks_comp)[0] - TP
    TN = 0  # note: check whether TN should really be set to 0

    performance = {
        'recall': TP / (TP + FN),
        'precision': TP / (TP + FP),
        'accuracy': (TP + TN) / (TP + FP + FN + TN),
        'f1_score': 2. * TP / (2. * TP + FP + FN),
    }

    return  idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance


def plot_nf_match_neurons_in_binary_masks(masks_gt, masks_comp, idx_tp_gt, idx_fn_gt, idx_tp_comp, idx_fp_comp,
                                          labels=None, cmap='viridis', Cn=None, levels=(.98,)):
    """The plotting part of the old nf_match_neurons_in_binary_masks() function.  Takes the outputs of that function."""
    lp, hp = np.nanpercentile(Cn, [5, 95])
    fig, axes = plt.subplots(ncols=2)
    for ax, (pos_comp, neg_gt) in zip(axes, [(idx_tp_comp, idx_tp_gt), (idx_fp_comp, idx_fn_gt)]):
        ax.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        for mm in masks_comp[pos_comp]:
            ax.contour(norm_nrg(mm), levels=levels, colors='w', linewidths=1)
        for mm in masks_gt[neg_gt]:
            ax.contour(norm_nrg(mm), levels=levels, colors='r', linewidths=1)
        ax.set_title('MATCHES')
        ax.set_axis('off')
    return fig


def norm_nrg(array):
    """threshold"""  # todo: correct documentation (function doesn't do any thresholding.)
    new_array = np.cumsum(np.sort(array, axis=None)[::-1] ** 2) # Check function
    new_array /= new_array.max()
    return new_array.reshape(array.shape, order='F')


def distance_masks(M_s, cm_s, max_dist, enclosed_thr=None):
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order,
    with matrix i compared with matrix i+1

    Parameters:
    ----------
    M_s: tuples of 1-D arrays
        The thresholded A matrices (masks) to compare, output of threshold_components

    cm_s: list of list of 2-ples
        the centroids of the components in each M_s

    max_dist: float
        maximum distance among centroids allowed between components. This corresponds to a distance
        at which two components are surely disjoined
        
    enclosed_thr: float
        if not None set distance to at most the specified value when ground truth is a subset of inferred 

    Returns:
    --------
    D_s: list of matrix distances

    Raise:
    ------
    Exception('Nan value produced. Error in inputs')
    """
    D_s = []
    for gt_comp, test_comp, cmgt_comp, cmtest_comp in tqdm(zip(M_s[:-1], M_s[1:], cm_s[:-1], cm_s[1:])):

        #the number of components for each
        nb_gt = np.shape(gt_comp)[-1]
        nb_test = np.shape(test_comp)[-1]
        D = np.ones((nb_gt, nb_test))

        cmgt_comp = np.array(cmgt_comp)
        cmtest_comp = np.array(cmtest_comp)
        if enclosed_thr is not None:
            gt_val = gt_comp.T.dot(gt_comp).diagonal()

        for i in range(nb_gt):
            k = gt_comp[:, np.repeat(i, nb_test)] + test_comp  # k is correlation matrix of this neuron to every other of the test

            for j  in range(nb_test): #for each components on the tests
                dist = np.linalg.norm(cmgt_comp[i]-cmtest_comp[j])  # we compute the distance of this one to the other ones
                union = k[:, j].sum()   # union matrix of the i-th neuron to the jth one
                if dist >= max_dist or union:  # if we don't have even a union this is pointless
                    D[i, j] = 1
                    continue

                intersection= np.array(gt_comp[:,i].T.dot(test_comp[:,j]).todense()).squeeze()  # product of the two elements' matrices; we multiply the boolean values from the jth omponent to the ith
                d = 1 - 1. * intersection / (union - intersection)
                if enclosed_thr is None:
                    #intersection is removed from union since union contains twice the overlapping area
                    #having the values in this format 0-1 is helpful for the hungarian algorithm that follows
                    D[i,j] = d
                elif intersection == gt_val[i]:
                    D[i, j] = min(d, 0.5)

        if np.any(np.isnan(D)):
            raise Exception('Nan value produced. Error in inputs')
        D_s.append(D)

    return D_s


def find_matches(distance_matrix):
    #todo: todocument
    if np.any(np.isnan(distance_matrix)):
        raise ValueError('Distance Matrix may not contain NaN values.')

    matches = [optimize.linear_sum_assignment(D) for D in tqdm(distance_matrix)]
    costs = [D[indices] for D, indices in zip(distance_matrix, matches)]
    return matches, costs


def link_neurons(matches, costs, max_cost=0.6, min_FOV_present=None):
    """
    Link neurons from different FOVs given matches and costs obtained from the hungarian algorithm

    Parameters:
    ----------
    matches: lists of list of tuple
        output of the find_matches function

    costs: list of lists of scalars
        cost associated to each match in matches

    max_cost: float
        maximum allowed value of the 1- intersection over union metric    

    min_FOV_present: int
        number of FOVs that must consequently contain the neuron starting from 0. If none 
        the neuro must be present in each FOV

    Returns:
    --------
    neurons: list of arrays representing the indices of neurons in each FOV

    """
    if min_FOV_present is None:
        min_FOV_present = len(matches)

    neurons = []
    for neuron_idx in range(len(matches[0][0])):
        neuron = [neuron_idx]
        for (rows, cols), cost in zip(matches, costs):
            m_neur = np.where(rows == neuron_idx)[0].squeeze()
            if m_neur.size > 0 and cost[m_neur] <= max_cost:
                neuron.append(cols[m_neur])
            else:
                break
        if len(neuron) > min_FOV_present:
            neurons.append(neuron)

    neurons = np.array(neurons).T
    return neurons




def extract_binary_masks_blob(A, neuron_radius, dims, num_std_threshold=1, minCircularity=0.5, minInertiaRatio=0.2,
                              minConvexity=.8):
    """
    Function to extract masks from data. It will also perform a preliminary selection of good masks based on criteria like shape and size

    Parameters:
    ----------
    A: sparse matris
        contains the components as outputed from the CNMF algorithm

    neuron_radius: float
        neuronal radius employed in the CNMF settings (gSiz)

    num_std_threshold: int
        number of times above iqr/1.349 (std estimator) the median to be considered as threshold for the component

    minCircularity: float
        parameter from cv2.SimpleBlobDetector

    minInertiaRatio: float
        parameter from cv2.SimpleBlobDetector

    minConvexity: float
        parameter from cv2.SimpleBlobDetector

    Returns:
    --------
    masks: np.array

    pos_examples:

    neg_examples:

    """
    params = cv2.SimpleBlobDetector_Params()
    params.minCircularity = minCircularity
    params.minInertiaRatio = minInertiaRatio
    params.minConvexity = minConvexity

    # Change thresholds
    params.blobColor = 255
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep= 3
    params.minArea = np.pi * ((neuron_radius * .75) ** 2)

    params.filterByColor = True
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True

    detector = cv2.SimpleBlobDetector_create(params)

    masks_ws, pos_examples, neg_examples = [], [], []
    for count, comp in tqdm(enumerate(A.tocsc()[:].T)):

        comp_d = np.array(comp.todense())
        gray_image = np.reshape(comp_d, dims, order='F')
        gray_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image)) * 255
        gray_image = gray_image.astype(np.uint8)

        # segment using watershed
        markers = np.zeros_like(gray_image)
        elevation_map = filters.sobel(gray_image)
        thr_1 = np.percentile(gray_image[gray_image > 0], 50)
        thr_2 = thr_1 + num_std_threshold * np.diff(np.percentile(gray_image[gray_image > 0], (25, 75))) / 1.35
        markers[gray_image < thr_1] = 1
        markers[gray_image > thr_2] = 2
        edges = watershed(elevation_map, markers) - 1

        # only keep largest object
        label_objects, nb_labels = ndimage.label(edges)
        sizes = np.bincount(label_objects.ravel())

        if len(sizes) > 1:
            idx_largest = np.argmax(sizes[1:]) + 1
            edges = label_objects == idx_largest
            edges = ndimage.binary_fill_holes(edges)
        else:  # empty component
            edges = np.zeros_like(edges)

        masks_ws.append(edges)

        if len(detector.detect((edges * 200.).astype(np.uint8))) > 0:
            pos_examples.append(count)
        else:
            neg_examples.append(count)

    return np.array(masks_ws),np.array(pos_examples),np.array(neg_examples)


#%%
def extract_binary_masks_blob_parallel(A,  neuron_radius,dims,num_std_threshold=1, minCircularity= 0.5,
                                       minInertiaRatio = 0.2,minConvexity = .8,dview=None):
    #todo todocument

    pars=[]    
    for a in range(A.shape[-1]):
        pars.append([A[:,a],  neuron_radius,dims,num_std_threshold, minCircularity, minInertiaRatio,minConvexity])
    if dview is not None:
        res=dview.map_sync(extract_binary_masks_blob_parallel_place_holder,pars)
    else:
        res=list(map(extract_binary_masks_blob_parallel_place_holder,pars))

    masks=[]
    is_pos=[]
    is_neg=[]
    for r in res:
        masks.append(np.squeeze(r[0]))
        is_pos.append(r[1])
        is_neg.append(r[2])

    masks=np.dstack(masks).transpose([2,0,1])
    return  masks,is_pos,is_neg


#%%   
def extract_binary_masks_blob_parallel_place_holder(pars):
    A,  neuron_radius, dims, num_std_threshold , minCircularity , minInertiaRatio ,minConvexity =pars
    masks_ws,pos_examples,neg_examples = extract_binary_masks_blob(A,  neuron_radius,
                                                                   dims,num_std_threshold=num_std_threshold, minCircularity= 0.5,
                                                                   minInertiaRatio = minInertiaRatio,minConvexity = minConvexity)
    return masks_ws,len(pos_examples),len(neg_examples)

#%%
def extractROIsFromPCAICA(spcomps, numSTD=4, gaussiansigmax=2 , gaussiansigmay=2,thresh=None):
    """
    Given the spatial components output of the IPCA_stICA function extract possible regions of interest

    The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing

    Parameters:
    -----------

    spcompomps, 3d array containing the spatial components

    numSTD: number of standard deviation above the mean of the spatial component to be considered signiificant
    """
    maskgrouped, allMasks = [], []
    for spcomp in spcomps:
        comp = filters.gaussian_filter(spcomp, [gaussiansigmay, gaussiansigmax])  # note: gaussian_filter() is deprecated.

        # get both positive and negative large magnitude pixels
        iqr = np.percentile(comp, 75) - np.percentile(comp, 25)
        minCompValuePos = thresh if thresh is not None else np.median(comp) + numSTD * iqr / 1.35
        minCompValueNeg = -thresh if thresh is not None else np.median(comp) - numSTD * iqr / 1.35
        compabspos = comp * (comp > minCompValuePos) - comp * (comp < minCompValueNeg)

        labeledpos, n = ndimage.label(compabspos > 0, np.ones((3, 3)))
        maskgrouped.append(labeledpos)
        for jj in range(1, n + 1):
            tmp_mask= np.asarray(labeledpos == jj)
            allMasks.append(tmp_mask)

    return allMasks, maskgrouped


def detect_duplicates(file_name,dist_thr = 0.1, FOV = (512,512)):
    """
    Removes duplicate ROIs from file file_name

    Parameters:
    -----------
        file_name:  .zip file with all rois

        dist_thr:   distance threshold for duplicate detection

        FOV:        dimensions of the FOV
    
    Returns:
    --------
        duplicates  : list of indeces with duplicate entries
        
        ind_keep    : list of kept indeces
        
    """
    rois = nf_read_roi_zip(file_name,FOV)
    cm = [ndimage.center_of_mass(mm) for mm in rois]
    sp_rois = sparse.csc_matrix(np.reshape(rois,(rois.shape[0],np.prod(FOV))).T)
    D = distance_masks([sp_rois,sp_rois],[cm,cm], 10)[0]
    np.fill_diagonal(D,1)
    indeces = np.where(D<dist_thr)      # pairs of duplicate indeces

    ind = list(np.unique(indeces[1][indeces[1]>indeces[0]]))
    ind_keep = list(set(range(D.shape[0]))-set(ind))
    duplicates = list(np.unique(np.concatenate((indeces[0],indeces[1]))))
    
    return duplicates, ind_keep





    # %% threshold and remove spurious components
    # def threshold_components(A_s,shape,min_size=5,max_size=np.inf,max_perc=.3):
    #    """
    #    Threshold components output of a CNMF algorithm (A matrices)
    #
    #    Parameters:
    #    ----------
    #
    #    A_s: list
    #        list of A matrice output from CNMF
    #
    #    min_size: int
    #        min size of the component in pixels
    #
    #    max_size: int
    #        max size of the component in pixels
    #
    #    max_perc: float
    #        fraction of the maximum of each component used to threshold
    #
    #
    #    Returns:
    #    -------
    #
    #    B_s: list of the thresholded components
    #
    #    lab_imgs: image representing the components in ndimage format
    #
    #    cm_s: center of masses of each components
    #    """
    #
    #    B_s=[]
    #    lab_imgs=[]
    #
    #    cm_s=[]
    #    for A_ in A_s:
    #        print('*')
    #        max_comps=A_.max(0).todense().T
    #        tmp=[]
    #        cm=[]
    #        lim=np.zeros(shape)
    #        for idx,a in enumerate(A_.T):
    #            #create mask by thresholding to 50% of the max
    #            print(idx)
    #            mask=np.reshape(a.todense()>(max_comps[idx]*max_perc),shape)
    #            label_im, nb_labels = ndimage.label(mask)
    #            sizes = ndimage.sum(mask, label_im, list(range(nb_labels + 1)))
    #            l_largest=(label_im==np.argmax(sizes))
    #            cm.append(ndimage.measurements.center_of_mass(l_largest,l_largest))
    #            lim[l_largest] = (idx+1)
    #    #       #remove connected components that are too small
    #            mask_size=np.logical_or(sizes<min_size,sizes>max_size)
    #            if np.sum(mask_size[1:])>1:
    #                print(('removing ' + str( np.sum(mask_size[1:])-1) + ' components'))
    #            remove_pixel=mask_size[label_im]
    #            label_im[remove_pixel] = 0
    #
    #            label_im=(label_im>0)*1
    #
    #            tmp.append(label_im.flatten())
    #
    #
    #        cm_s.append(cm)
    #        lab_imgs.append(lim)
    #        B_s.append(csc.csc_matrix(np.array(tmp)).T)
    #
    #    return B_s, lab_imgs, cm_s

    # %% remove duplicate ROIs from manually annotated files


    #%%
#def get_roi_from_spatial_component(sp_comp, min_radius, max_radius, roughness=2, zoom_factor=1, center_range=2,z_step=1,thresh_iou=.4):
#     
#     
#     if sp_comp.ndim > 2:
#         center=center_of_mass(sp_comp[0])
#         roi=cell_magic_wand_3d_IOU(sp_comp,center,min_radius, max_radius, roughness=roughness, zoom_factor=zoom_factor, center_range=center_range, z_step=z_step,thresh_iou=thresh_iou)
#     else:            
#         center=center_of_mass(sp_comp)
#         roi=cell_magic_wand(sp_comp,center,min_radius, max_radius, roughness=roughness, zoom_factor=zoom_factor, center_range=center_range)
#     print sp_comp.shape,roi.shape
#     plt.subplot(1,2,1)
#     plt.cla()
#     if sp_comp.ndim > 2:
#         plt.imshow(sp_comp[0])
#     else:
#         plt.imshow(sp_comp)
#     plt.subplot(1,2,2)
#     plt.cla()
#     plt.imshow(roi)
##     plt.plot(center[1],center[0],'k.')
#     plt.pause(.3)
#     print 'Roi Size:' + str(np.sum(roi)) 
#     return roi
##%%     
#def get_roi_from_spatial_component_place_holder(pars):
#    sp_comp, min_radius, max_radius, roughness, zoom_factor, center_range,z_step,thresh_iou = pars
#    roi=get_roi_from_spatial_component(sp_comp,min_radius, max_radius, roughness=roughness, zoom_factor=zoom_factor, center_range=center_range,z_step=z_step,thresh_iou=thresh_iou)
#    return roi
##%%
#def get_roi_from_spatial_component_parallel(sp_comps,min_radius, max_radius, other_imgs=None,dview=None, roughness=2, zoom_factor=1, center_range=2,z_step=1,thresh_iou=.4):   
#
#    pars=[]
#    for sp_comp in sp_comps:
#        if other_imgs is None:
#            pars.append([sp_comp, min_radius, max_radius, roughness, zoom_factor, center_range,z_step,thresh_iou])
#        else:
#           
#            pars.append([np.dstack([sp_comp]+other_imgs).transpose(2,0,1), min_radius, max_radius, roughness, zoom_factor, center_range,z_step,thresh_iou])
#            
#        
#    if dview is not None:
#        rois=dview.map_sync(get_roi_from_spatial_component_place_holder,pars)
#        
#    else:
#        rois=map(get_roi_from_spatial_component_place_holder,pars)
#        
#    return rois        
