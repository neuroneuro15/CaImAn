#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult CaImAn documentation and 

"""
from __future__ import print_function

try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import numpy as np
# mpl.use('Qt5Agg')
import pylab as pl
pl.ion()
#%
import caiman as cm
from caiman import behavior

#%%

n_components = 3
m = cm.load('./test.tif')
m.play(gain=3.)
#%% extract optical flow and ry NMF on it
mask = behavior.select_roi(np.median(m[::100], 0), 1)[0] # select the portion of FOV interesting (could be all of it as well)
resize_fact = .5
num_std_mag_for_angle = .6
whole_field = True
only_magnitude = False
spatial_filter_, time_trace_, of_or = caiman.behavior.extract_motor_components_OF(m, n_components, mask = mask, resize_fact= resize_fact, only_magnitude = only_magnitude, max_iter = 1000, verbose = True, method_factorization ='nmf')
mags, dircts, dircts_thresh, spatial_masks_thrs = caiman.behavior.extract_magnitude_and_angle_from_OF(spatial_filter_, time_trace_, of_or, num_std_mag_for_angle = num_std_mag_for_angle, sav_filter_size =3, only_magnitude = only_magnitude)
#%% if you want to visualize optical flow
ms = [mask*fr for fr in m]
ms = np.dstack(ms)
ms = cm.Movie(ms.transpose([2, 0, 1]))
_ = caiman.behavior.compute_optical_flow(ms, do_show=True, polar_coord=True)

#%% spatial components
count = 0
for comp in spatial_filter_:
    count+=1
    pl.subplot(2,2,count)
    pl.imshow(comp)
#%% temporal components (magnitude and angle in polar coordinates)
count = 0
for magnitude, angle in zip(mags,dircts):
    count+=1
    pl.subplot(2,2,count)
    pl.plot(magnitude/np.nanmax(magnitude))
    angle_ = angle.copy()
    angle_[magnitude<np.std(magnitude)*1.5] = np.nan
    pl.plot(angle_/2/np.pi)