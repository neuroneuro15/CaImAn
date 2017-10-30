from __future__ import print_function
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import HTML, display
from matplotlib.colors import cnames
from matplotlib import animation, rc
from scipy.sparse import coo_matrix
from past.utils import old_div
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.motion_correction import MotionCorrect
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman import save_memmap_join
from caiman.mmapping import load_memmap
import scipy
import pylab as pl
import matplotlib as mpl
import matplotlib.cm as mcm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
#import select_file as sf_
from typing import Dict, Tuple, List
try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d, Spacer
    from bokeh.layouts import row, widgetbox, column
    bpl.output_notebook()
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

#from ..summary_images import local_correlations


######
class Context:
	def __init__(self, cluster):
		#setup cluster
		self.c = cluster[0]
		self.dview = cluster[1]
		self.n_processes = cluster[2]
		self.working_dir = ''
		self.working_mc_files = [] #str path
		self.working_cnmf_file = None  #str path

		self.mc_rig = []    #rigid mc results
		self.mc_nonrig = [] #non-rigid mc results

		self.YrDT = None # tuple (Yr, dims, T) numpy array (memmory mapped file)
		self.cnmf_results = [] #A, C, b, f, YrA, sn, idx_components
		self.cnmf_idx_components_keep = []  #result after filter_rois()
		self.cnmf_idx_components_toss = []
		#rest of properties
		self.cnmf_params = None # CNMF Params: Dict


#setup cluster
def start_procs():
	c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
	return c, dview, n_processes


#converts all files to tif for motion correction, unless they're already tif
'''def load_files(path,dsx=0.5,dsy=0.5,dst=1): #NOTE: This downsamples spatially
	files = glob.glob(fldr + '*.tif') + glob.glob(fldr + '*.tiff') + glob.glob(fldr + '*.avi')
	list_of_files = []
	for each_file in files:
		splitFile = os.path.splitext(each_file)
		tiff_file = each_file
		toTiff = cm.load(each_file)
		if splitFile[1] != '.tif' and splitFile[1] != '.tiff':
			tiff_file = splitFile[0] + "_dsx_" + dsx + "_dsy_" + dsy + "_dst_" + dst + '.tif'
			toTiff.save(tiff_file)
		tiff_file = splitFile[0] + "_dsx_" + dsx + "_dsy_" + dsy + "_dst_" + dst + '.tif'
		list_of_files.append(tiff_file)
		toTiff.resize(dsx,dsy,dst).save(tiff_file)
	return list_of_files'''

def load_files(fldr, print_values=False):
	if os.path.isfile(fldr):
		if print_values: print("Loaded: " + fldr)
		return [fldr]
	else:
		files = glob.glob(fldr + '*.tif') + glob.glob(fldr + '*.tiff') + glob.glob(fldr + '*.avi')
		if print_values:
			print("Loaded files:")
			for each_file in files:
				print(each_file)
		return files

#run motion correction
def run_mc(fnames, mc_params, dsfactors, rigid=True, batch=True):
	min_mov = 0
	mc_list = []
	new_templ = None
	counter = 0
	def resave_and_resize(each_file, dsfactors):
		toTiff = cm.load(each_file)
		tiff_file = each_file
		tiff_file = os.path.splitext(each_file)[0] + '.tif'
		if any(x < 1 for x in dsfactors):
			toTiff.resize(*dsfactors).save(tiff_file)
		else:
			toTiff.save(tiff_file)
		return tiff_file
	for each_file in fnames:
		#first convert AVI to TIFF and DOWNSAMPLE SPATIALLY
		tiff_file = each_file
		is_converted = False
		file_ext = os.path.splitext(each_file)[1]
		if (file_ext == '.avi' or any(x < 1 for x in dsfactors)):
			is_converted = True
			if file_ext != '.tif':
				print("Converting %s to TIF file format..." % (os.path.basename(tiff_file,)))
			if any(x < 1 for x in dsfactors):
				print("Downsampling by %s x, %s y, %s t (frames), resaving as temporary file" % dsfactors)
			tiff_file = resave_and_resize(each_file, dsfactors)
		#get min_mov
		if counter == 0:
			min_mov = np.array([cm.motion_correction.low_pass_filter_space(m_,mc_params['gSig_filt']) for m_ in cm.load(tiff_file, subindices=range(999))]).min()
			#min_mov = cm.load(tiff_file, subindices=range(400)).min()
			print("Min Mov: ", min_mov)
			print("Motion correcting: " + tiff_file)
			
	# TODO: needinfo how the classes works
		new_templ = None
		mc_mov = None
		bord_px_rig = None
		bord_px_els = None
		#setup new class object
		mc = MotionCorrect(tiff_file, min_mov, **mc_params)

		if rigid:
			print("Starting RIGID motion correction...")
			mc.motion_correct_rigid(save_movie=True, template = new_templ)
			new_templ = mc.total_template_rig
			mc_mov = cm.load(mc.fname_tot_rig)
			bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
		else:
			print("Starting NON-rigid motion correction...")
			mc.motion_correct_pwrigid(save_movie=True, template=new_templ, show_template=False)
			new_templ = mc.total_template_els
			mc_mov = cm.load(mc.fname_tot_els)
			bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)

		# TODO : needinfo
		#pl.imshow(new_templ, cmap='gray')
		#pl.pause(.1)
		mc_list.append(mc)
		#remove generated TIFF, if applicable
		if is_converted:
			os.remove(tiff_file)
		counter += 1
	clean_up() #remove log files
	if batch:
		print("Batch mode, combining files")
		combined_file = combine_mc_mmaps(mc_list,mc_params['dview'])
		#delete individual files
		for old_f in mc_list:
			if rigid:
				os.remove(old_f.fname_tot_rig)
			else:
				os.remove(old_f.fname_tot_els)
		return [combined_file]
	else:
		return mc_list


def combine_mc_mmaps(mc_list, dview):
	mc_names = [i.fname_tot_rig for i in mc_list]
	mc_mov_name = save_memmap_join(mc_names, base_name='mc_rig', dview=dview)
	print(mc_mov_name)
	return mc_mov_name

def tiff2mmap():
	fldr = '/Users/brandonbrown/Desktop/KhakhLab/MiniscopeProject/ForLabMtng/'
	files_tif = glob.glob(fldr + '*.tif')
	for f in files_tif:
		cm.load(f).resize(0.8,0.799,1).save(os.path.splitext(f)[0] + '.mmap') #should produce 192x300xFrames


def resize_mov(Yr, fx=0.521, fy=0.3325):
	t,h,w = Yr.shape
	newshape=(int(w*fy),int(h*fx))
	mov=[]
	print(newshape)
	for frame in Yr:
		mov.append(cv2.resize(frame,newshape,fx=fx,fy=fy,interpolation=cv2.INTER_AREA))
	return np.asarray(mov)

#clean up tif files if originals were not already tif
def clean_up_files():
	pass

def cnmf_run(fname: str, cnmf_params: Dict): #fname is a full path, mmap file
	#SETTINGS
	#gSig = 4   # gaussian width of a 2D gaussian kernel, which approximates a neuron
	#gSiz = 12  # average diameter of a neuron
	#min_corr = 0.8 #0.8 default   ([0.65, 3] => 91 kept neurons; good specificity, poor sensitivity in periphery)
	#min_pnr = 1.1 #10 default
	# If True, the background can be roughly removed. This is useful when the background is strong.
	center_psf = True
	Yr, dims, T = cm.load_memmap(fname)
	print(dims, " ; ", T)
	Yr = Yr.T.reshape((T,) + dims, order='F')
	#configs
	cnm = cnmf.CNMF(**cnmf_params)
	cnm.fit(Yr)
	'''	cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=20, gSig=(5, 5), gSiz=(5, 5), merge_thresh=.8,
				p=1, dview=dview, tsub=1, ssub=1,p_ssub=2, Ain=None, rf=(25, 25), stride=(15, 15),
				only_init_patch=True, gnb=5, nb_patch=3, method_deconvolution='oasis',
				low_rank_background=False, update_background_components=False, min_corr=min_corr,
				min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
				ring_size_factor=1.5, center_psf=True)'''
	#get results
	A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
	idx_components = np.arange(A.shape[-1])
	clean_up() #remove log files
	return A,C, b, f, YrA, sn, idx_components


def plot_contours(YrDT: Tuple, cnmf_results: Tuple, cn_filter):
	Yr, dims, T = YrDT
	Yr = np.rollaxis(np.reshape(Yr, dims + (T,), order='F'), 2)
	A, C, b, f, YrA, sn, idx_components = cnmf_results
	pl.figure()
	crd = cm.utils.visualization.plot_contours(A.tocsc()[:, idx_components], cn_filter, thr=.9)
	'''
	#%%
	plt.imshow(A.sum(-1).reshape(dims, order='F'), vmax=200)
	#%%
	'''
	cm.utils.visualization.view_patches_bar(
		YrA, coo_matrix(A.tocsc()[:, idx_components]), C[idx_components],
		b, f, dims[0], dims[1], YrA=YrA[idx_components], img=cn_filter)

def filter_rois(YrDT: Tuple, cnmf_results: Tuple):
	Yr, dims, T = YrDT
	A, C, b, f, YrA, sn, idx_components_orig = cnmf_results
	final_frate = 20# approx final rate  (after eventual downsampling )
	Npeaks = 10
	traces = None
	try:
		traces = C + YrA
	except ValueError:
		traces = C + YrA.T
	#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
	#        traces_b=np.diff(traces,axis=1)
	Y = np.reshape(Yr, dims + (T,), order='F')
	fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = cm.components_evaluation.evaluate_components(
		Y, traces, A, C, b, f, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

	idx_components_r = np.where(r_values >= .5)[0]
	idx_components_raw = np.where(fitness_raw < -40)[0]
	idx_components_delta = np.where(fitness_delta < -20)[0]

	idx_components = np.union1d(idx_components_r, idx_components_raw)
	idx_components = np.union1d(idx_components, idx_components_delta)
	idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

	print(('Keeping ' + str(len(idx_components)) +
		   ' and discarding  ' + str(len(idx_components_bad))))
	return idx_components, idx_components_bad


def corr_img(Yr: np.ndarray, gSig: int, center_psf :bool):
	# show correlation image of the raw data; show correlation image and PNR image of the filtered data
	cn_raw = cm.summary_images.max_correlation_image(Yr, swap_dim=False, bin_size=3000) #default 3000
	#%% TAKES MEMORY!!!
	cn_filter, pnr = cm.summary_images.correlation_pnr(
		Yr, gSig=gSig, center_psf=center_psf, swap_dim=False)
	plot_ = plt.figure(figsize=(10, 5))
	#%%
	for i, (data, title) in enumerate(((Yr.mean(0), 'Mean image (raw)'),
									   (Yr.max(0), 'Max projection (raw)'),
									   (cn_raw[1:-1, 1:-1], 'Correlation (raw)'),
									   (cn_filter, 'Correlation (filtered)'),
									   (pnr, 'PNR (filtered)'),
									   (cn_filter * pnr, 'Correlation*PNR (filtered)'))):
		plt.subplot(2, 3, 1 + i)
		plt.imshow(data, cmap='jet', aspect='equal')
		plt.axis('off')
		plt.colorbar()
		plt.title(title)
	return cn_raw, cn_filter, pnr, plot_
'''
def pick_thresholds():
	# pick thresholds
	fig = plt.figure(figsize=(10, 4))
	plt.axes([0.05, 0.2, 0.4, 0.7])
	im_cn = plt.imshow(cn_filter, cmap='jet')
	plt.title('correlation image')
	plt.colorbar()
	plt.axes([0.5, 0.2, 0.4, 0.7])
	im_pnr = plt.imshow(pnr, cmap='jet')
	plt.title('PNR')
	plt.colorbar()

	s_cn_max = Slider(plt.axes([0.05, 0.01, 0.35, 0.03]), 'vmax',
					  cn_filter.min(), cn_filter.max(), valinit=cn_filter.max())
	s_cn_min = Slider(plt.axes([0.05, 0.07, 0.35, 0.03]), 'vmin',
					  cn_filter.min(), cn_filter.max(), valinit=cn_filter.min())
	s_pnr_max = Slider(plt.axes([0.5, 0.01, 0.35, 0.03]), 'vmax',
					   pnr.min(), pnr.max(), valinit=pnr.max())
	s_pnr_min = Slider(plt.axes([0.5, 0.07, 0.35, 0.03]), 'vmin',
					   pnr.min(), pnr.max(), valinit=pnr.min())
'''
def save_data():
	#TODO: should include contour plot somehow
	np.savez(folder2 + 'analysisResults2.npz', A=A, C=C)

def save_denoised_avi(data, dims, idx_components_keep):
	A, C, b, f, YrA, sn, idx_components = data
	idx_components = idx_components_keep
	x = None
	if type(A) != np.ndarray:
		x = cm.movie(A.tocsc()[:, idx_components].dot(C[idx_components, :])).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
	else:
		x = cm.movie(A[:, idx_components].dot(C[idx_components, :])).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
	x.save("denoised.avi")
	print("Saved denoised.avi")

def load_data():
	#Need to import contour plot somehow
	#np.load()...
	pass

def view_edit_results(Yr, A, C, b, f, d1, d2, YrA = None, image_neurons=None, thr=0.99, denoised_color=None,cmap='jet'):
    """
    Interactive plotting utility for ipython notebook

    Parameters:
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    YrA:   np.ndarray
        ROI filtered residual as it is given from update_temporal_components
        If not given, then it is computed (K x T)        

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    cmap: string
        name of colormap (e.g. 'viridis') used to plot image_neurons
    """
    colormap = mcm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.ravel(np.power(A,2).sum(0)) if type(A) == np.ndarray else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                   (A.T * np.matrix(Yr) -
                    (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                    A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA
            

    x = np.arange(T)
    z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = cm.utils.visualization.get_contours(A, (d1, d2), thr)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=z[:, 0], y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=z.T, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2, source2_=source2_), code="""
            var data = source.get('data')
            var data_ = source_.get('data')
            var f = cb_obj.get('value')-1
            x = data['x']
            y = data['y']
            y2 = data['y2']

            for (i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.get('data');
            var data2 = source2.get('data');
            c1 = data2['c1'];
            c2 = data2['c2'];
            cc1 = data2_['cc1'];
            cc2 = data2_['cc2'];

            for (i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.trigger('change')
            source.trigger('change')
        """)

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)

    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)

    deleted_rois_text = bokeh.models.TextInput(title="Deleted ROIs")


    del_btn_callback = CustomJS(args=dict(roitxtbox=deleted_rois_text, slider=slider), code="""
    		var slider_val = slider.value;
    		var slider_val_str = ' ' + Math.floor(slider_val);
    		roitxtbox.value += slider_val_str;
    	""")
    del_roi_btn = bokeh.models.Button(label="Delete ROI", callback=del_btn_callback, width=100)
    #deleted_rois_array = [int(x) for x in deleted_rois_text.value.strip().split(' ')]
    #del_roi_btn.on_click(remove_roi_event)
    #var filetext = 'name,income,years_experience\n';
    #print(C.shape) #219x2000
    rois_source = ColumnDataSource(data=dict(c=C))
    dl_code = """
		var data_ = roi_data.data;
		var data = data_['c'];
		var rows = %s;
		var cols = %s;
		var remove_rois = deld_rois.value.split(' ').map(Number).map(function(value) { return value -1; });
		console.log(remove_rois);
		//console.log(rows);
		//console.log(cols);
		var filetext = '';
		//iterate over each row
		for (i=0; i < rows; i++) {
			if (remove_rois.includes(i)) { continue; }
			var start = (i * cols);
			var stop = start + cols;
		    var currRow = [i+1,data.slice(start, stop).toString().concat('\\n')];
		    var joined = currRow.join();
		    filetext = filetext.concat(joined);
		}

		var filename = 'data_result.csv';
		var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

		//addresses IE
		if (navigator.msSaveBlob) {
		    navigator.msSaveBlob(blob, filename);
		}

		else {
		    var link = document.createElement("a");
		    link = document.createElement('a')
		    link.href = URL.createObjectURL(blob);
		    link.download = filename
		    link.target = "_blank";
		    link.style.visibility = 'hidden';
		    link.dispatchEvent(new MouseEvent('click'))
		}
	""" % (C.shape[0], C.shape[1])


    dl_data_btn = bokeh.models.Button(label="Download Data", \
    	callback=CustomJS(args=dict(roi_data=rois_source, deld_rois=deleted_rois_text), code=dl_code), width=100)

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)
    #slider_box = row(slider)
    #rest_box = row(del_roi_btn, deleted_rois_text, dl_data_btn)
    spac1 = Spacer(width=50, height=100)
    #spac2 = Spacer(width=100, height=100)
    t = bpl.show(bokeh.layouts.layout([[row(slider, del_roi_btn, spac1, deleted_rois_text, dl_data_btn)], \
        [row(plot1, plot)]]), notebook_handle=True)

    return Y_r

#Use Matplotlib's animation ability to play embedded HTML5 video of Numpy array
#Need FFMPEG installed: brew install ffmpeg
def play_movie(movie, interval=50, blit=True, cmap='gist_gray', vmin=None, vmax=None):
	frames = movie.shape[0]
	fig = plt.figure()
	im = plt.imshow(movie[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
	def updatefig(i):
		im.set_array(movie[i,:,:])
		return im,
	anim = animation.FuncAnimation(fig, updatefig, frames=frames, interval=50, blit=blit)
	return HTML(anim.to_html5_video())

def remove_roi():
	pass

def download_data():
	pass

def clean_up(stop_server=False):
	if stop_server: cm.stop_server()
	log_files = glob.glob('*_LOG_*')
	for log_file in log_files:
		os.remove(log_file)