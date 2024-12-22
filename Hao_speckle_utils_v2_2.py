#!/usr/bin/python3

import numpy as np
import os, sys
import time
import itertools
import h5py
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm
import tifffile as tf
import sgolay2
from skimage.feature import peak_local_max


def pr_keys(d, level=0):
    try:
        k_lst = d.keys()
        for k in k_lst:
            print("\t"*level+f"-{k}")
            pr_keys(d[k], level=level+1)
    except:
        pass


def getSpecFile(fname, fpath=''):
   '''
   To get detailed info from sf, one could type
       help(sf)
   Or
       dir(sf)
   in the Jupyter notebook/lab
  
    |  SpecFileDataSource(nameInput)
    | 
    |  getDataObject(self, key, selection=None)
    |      Parameters:
    |      * key: key to be read from source. It is a string
    |            using the following formats:
    |     
    |          "s.o": loads all counter values (s=scan number, o=order)
    |            - if ScanType==SCAN: in a 2D array (mot*cnts)
    |            - if ScanType==MESH: in a 3D array (mot1*mot2*cnts)
    |            - if ScanType==MCA: single MCA in 1D array (0:channels)
    |     
    |          "s.o.n": loads a single MCA in a 1D array (0:channels)
    |            - if ScanType==NMCA: n is the MCA number from 1 to N
    |            - if ScanType==SCAN+MCA: n is the scan point number (from 1)
    |            - if ScanType==MESH+MCA: n is the scan point number (from 1)
    |     
    |          "s.o.p.n": loads a single MCA in a 1D array (0:channels)
    |            - if ScanType==SCAN+NMCA:
    |                    p is the point number in the scan
    |                    n is the MCA device number
    |            - if ScanType==MESH+MCA:
    |                    p is first motor index
    |                    n is second motor index
    |     
    |          "s.o.MCA": loads all MCA in an array
    |            - if ScanType==SCAN+MCA: 2D array (pts*mca)
    |            - if ScanType==NMCA: 2D array (mca_det*mca)
    |            - if ScanType==MESH+MCA: 3D array (pts_mot1*pts_mot2*mca)
    |            - if ScanType==SCAN+NMCA: 3D array (pts_mot1*mca_det*mca)
    |            - if ScanType==MESH+NMCA:
    |                    creates N data page, one for each MCA device,
    |                    with a 3D array (pts_mot1*pts_mot2*mca)
    | 
    |  getKeyInfo(self, key)
    |      If key given returns information of a perticular key.
    | 
    |  getSourceInfo(self)
    |      Returns information about the specfile object created by
    |      the constructor to give application possibility to know about
    |      it before loading.
    |      Returns a dictionary with the key "KeyList" (list of all available keys
    |      in this source). Each element in "KeyList" has the form 'n1.n2' where
    |      n1 is the scan number and n2 the order number in file starting at 1.
    | 
    |  isUpdated(self, sourceName, key)
    | 
    |  refresh(self)
    |
    |  Data and other attributes defined here:
    | 
    |  Error = 'SpecFileDataError'  
    '''
   if fpath=='':
       fpath = os.getcwd()
  
   filePath = os.path.abspath(os.path.join(fpath, fname))
   sf = SpecFileDataSource.SpecFileDataSource(filePath)
   return sf
 

def getSpecScan(sf, scanID):
   # In PyMca, the scans are labeled as '15.1' instead of '15'
   scan = sf.getDataObject(str(scanID)+'.1')
   scanKeys = scan.getInfo()['LabelNames']
   scanData = scan.getData()
   scanData = pd.DataFrame(scanData, columns=scanKeys)
   return scanData


def plot_2d(img, vmin=None, vmax=None, origin='lower', **kwargs):
    if not vmin:
        vmin = np.min(img)
    if not vmax:
        vmin = np.max(img)
    fig = go.Figure(data=go.Heatmap(z=img,
                                    zmin=vmin,
                                    zmax=vmax,
                                    colorscale = 'jet',
                                    **kwargs))
    fig.upate_layout(height=600,
                      width=600)
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    if origin == 'upper':
        fig.update_layout(yaxis=dict(autorange='reversed'))
    return fig


###############################################
# misc helper functions
###############################################    
      
def sizeof_fmt(num, suffix='B'):
    '''
    Covert size to human-friendly format
    --------
    Parameters:
    num : int, number of size
    suffix : str, suffix of output 
    --------
    Returns:
    str
    '''
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.2f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)


def n_in_m_bins(n, m):
    d, r = divmod(n, m)
    res = [[m*j+i for j in range(d)] for i in range(m)]
    for i in range(r):
        res[i].append(res[i][-1]+m)
    return res


def n_in_m_bins_seq(n, m):
    d, r = divmod(n, m)
    res_1 = [[j+i*(d+1) for j in range(d+1)] for i in range(r)]
    res_2 = [[(d+1)*r+j+i*d for j in range(d)] for i in range(m-r)]
    return res_1 + res_2


def timeit(func):
    '''
    Decorator counts time for running a function
    '''
    def inner(*args, **kwargs):
        ts = time.perf_counter()
        result = func(*args, **kwargs)
        te = time.perf_counter()
        if 'verbose' not in kwargs or kwargs['verbose'] == True:
            print(f'{func.__name__} finished in {te-ts:.3f} seconds')
        return result
    return inner


def find_cycle(vs, i_cycle=1, form='sin'):
    if form == 'sin':
        wave = fit_sin(np.arange(len(vs)), vs)
        i_start = round((i_cycle*2*np.pi-0.5*np.pi*(np.sign(wave['amp'])-1)-wave['phase'])/wave['omega'])
        i_end = i_start + wave['period']
    if form == 'squ':
        diff = np.array([vs[i]-vs[i-1] for i in range(1, len(vs))])
        starts = np.where(diff>0)
        i_start = starts[0][i_cycle]
        i_end = i_start  + wave['period']
    return i_start, i_end
    

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

###############################################
# ROI filter
###############################################

def line_roi(img, cen_pix=[], cut='H'):
    '''
    speckle in 2D:
    0d: y
    1d: x
    cut: can be 'H' or 'V'
    '''
    if cen_pix==[]:
        ycen = (img.shape[0]-1)//2
        xcen = (img.shape[1]-1)//2
    else:
        xcen = cen_pix[0]
        ycen = cen_pix[1]
    if cut=='H':
        roi = img[ycen, :]
    if cut=='V':
        roi = img[:, xCen]
    return roi


def box_roi(img, box):
    '''
    speckle in 2D:
    0d: y
    1d: x
    box = [xmin, xmax, ymin, ymax]
    '''
    if len(img.shape) == 2:
        return img[box[2]:box[3], box[0]:box[1]]
    if len(img.shape) == 3:
        return img[:, box[2]:box[3], box[0]:box[1]]

def cir_roi(img, rmin, rmax, cen_pix=[]):
    '''
    speckle in 2D:
    0d: y
    1d: x
    '''
    if cen_pix==[]:
        ycen = (img.shape[0]-1)/2
        xcen = (img.shape[1]-1)/2
    else:
        xcen = cen_pix[0]
        ycen = cen_pix[1]

    yline = np.arange(img.shape[0])-ycen
    xline = np.arange(img.shape[1])-xcen
    xmesh, ymesh = np.meshgrid(xline, yline)
    radius = np.sqrt(xmesh**2+ymesh**2)

    maskmin = np.where(radius>rmin, 1, np.nan)
    maskmax = np.where(radius<rmax, 1, np.nan)
    mask = maskmin*maskmax
    # Apply masks and generate waterfall
    img_ma = img * mask
    return img_ma.flatten()


def ellipse_filter(xys, a, b, phi, eps, cen_pix):
    x_rot = (xys[0]-cen_pix[0]) * np.cos(phi) - (xys[1]-cen_pix[1]) * np.sin(phi)
    y_rot = (xys[0]-cen_pix[0]) * np.sin(phi) + (xys[1]-cen_pix[1]) * np.cos(phi)
    ellipse = x_rot**2/a**2 + y_rot**2/b**2
    return np.logical_and(ellipse > 1 - eps, ellipse < 1 + eps)


def elliptical_roi(waterfall, a, b, phi, eps, cen_pix):
    x = np.linspace(0, 1029, 1030)
    y = np.linspace(0, 513, 514)
    xs, ys = np.meshgrid(x, y)
    xys = np.array([xs.flatten(), ys.flatten()])
    mask = ellipse_filter(xys, a, b, phi, eps, cen_pix)
    roi = []
    for img in waterfall:
        img = img.flatten()
        roi.append(img[mask])
    return np.array(roi)


def show_roi(img, roi, ax=None, vmax=None, *arg, **kwargs):
    if ax is None:
        ax = plt.gca()
    if vmax is None:
        vmax = 0.5*np.nanmax(img)
    x_min, x_max, y_min, y_max = roi
    ax.imshow(img, origin='lower', vmin=0, vmax=vmax, *arg, **kwargs)
    ax.plot([i for i in range(x_min, x_max+1)], [y_min for i in range(x_min, x_max+1)], 'r-')
    ax.plot([i for i in range(x_min, x_max+1)], [y_max for i in range(x_min, x_max+1)], 'r-')
    ax.plot([x_min for i in range(y_min, y_max+1)], [i for i in range(y_min, y_max+1)], 'r-')
    ax.plot([x_max for i in range(y_min, y_max+1)], [i for i in range(y_min, y_max+1)], 'r-')
    return ax


def smooth_sg(img, roi=None, window_size=7, poly_order=5):
    if roi is None:
        return sgolay2.SGolayFilter2(window_size=window_size, poly_order=poly_order)(img)
    else:
        x_min, x_max, y_min, y_max = roi
        img_roi = box_roi(img, roi)
        img_pad = box_roi(img, [x_min-window_size, x_max+window_size, y_min-window_size, y_max+window_size])
        sgfit = sgolay2.SGolayFilter2(window_size=window_size, poly_order=poly_order)(img_pad)
        bkgd = sgfit[window_size:-window_size, window_size:-window_size]
        return bkgd
    

def find_local_max(img, min_distance=5, th_factor=0.2):
    img_smooth = smooth_sg(img, window_size=21, poly_order=3)
    yx = peak_local_max(img_smooth, min_distance=5, threshold_abs=th_factor*np.max(img_smooth))
    return yx


def norm_speckles_sg(img, roi, window_size=7, poly_order=5):
    img_roi = np.array(box_roi(img, roi), dtype=float)
    bkgd_roi = smooth_sg(img, roi, window_size=window_size, poly_order=poly_order)
    return np.divide(img_roi-bkgd_roi, (bkgd_roi+img_roi), out=np.zeros_like(img_roi), where=bkgd_roi!=0)


def remove_bad_pixs(img, bad_pixs):
    if len(img.shape) == 2:
        for y, x in bad_pixs:
            img[y, x] = np.nan
    if len(img.shape) == 3:
        for y, x in bad_pixs:
            img[:, y, x] = np.nan
    return img


###############################################
# h5 I/O
###############################################

def h5_stats(run_num, 
             prefix='xpplx5019_Run', 
             key='jungfrau1M/ROI_0_area',
             roi_fn=box_roi,
             roi_params=[],
             save=True):
    f_name = f'{prefix}{run_num:04d}'
    file = os.path.join(DATA_PATH, f_name + '.h5')
    h = h5py.File(file)
    imgs = h[key]
    
    # need to change according to data format
    n_imgs, n_detect, Truen_x, n_y = imgs.shape
    
    w_list = []
    s_list = []
    t_range = tqdm(range(n_imgs))
    t_range.set_description(f"Reading frames from h5 file")
    for i in t_range:
        # need to adjust according to the detector alignment
        img = np.concatenate(imgs[i])
        
        roi = roi_fn(img, roi_params)
        roi_mean = np.mean(roi)
        roi_var = np.var(roi)
        w_list.append(roi)
        s_list.append([roi_mean, roi_var])
        #stats.append(gen_stats())
    waterfall = np.array(w_list)
    stats = np.array(s_list)
    if save:
        waterfall_name =  f'wf_Run{run_num}_{roi_fn.__name__}{roi_params}.csv'
        stats_name = f'sts_Run{run_num}_{roi_fn.__name__}{roi_params}.csv'
        np.savetxt(os.path.join(SAVE_PATH, waterfall_name), 
                   waterfall.reshape(waterfall.shape[0], waterfall.shape[1]*waterfall.shape[2]))
        np.savetxt(os.path.join(SAVE_PATH, stats_name), stats)
    return waterfall, stats


###############################################
# g2 curve fitting
###############################################

def siegert(t, c, tau):
    '''
    Siegert function for fitting autocorrelation vs tau
    g2 = 1 + beta * exp(- 2 * tau/ tau_c)
    params: beta, tau_c
    '''
    return 1 + c * np.exp(-t/tau)

def stretched_siegert(t, c, tau, beta):
    return 1 + c * np.exp(-(t/tau)**beta)


def fit_g2(g2, ts=[], fit_range='full', func=stretched_siegert, params_init=(1, 30, 1), plot=True):
    '''
    Curve fitting for autocorrelation
    --------
    Parameters:
    g2: 1D numpy array, autocorrelation function
    ts: list, list of tau, should have the same dim as g2
    fit_range: [i_min, i_max], specify it when you want to fit part of data
    params_init: list, list of initial guess of parameters
    --------
    '''
    if not ts:
        ts = [i for i in range(g2.shape[0])]
    if fit_range != 'full':
        t_lo, t_hi = fit_range
        ts_fit = ts[t_lo:t_hi]
        g2_fit = g2[t_lo:t_hi]
    else:
        ts_fit = ts
        g2_fit = g2
    params_fit, cv = scipy.optimize.curve_fit(func, ts_fit, g2_fit, params_init)

    fit_curve = np.array([func(t, *params_fit) for t in ts_fit])
    squared_diffs = np.square(g2_fit - fit_curve)
    squared_diffs_from_mean = np.square(g2_fit - np.mean(g2_fit))
    r_squared = 1 - np.sum(squared_diffs) / np.sum(squared_diffs_from_mean)
    print(f'Fitting function: {func.__name__}\n{func.__doc__}')
    print(f'params={params_fit}')
    print(f"R² = {r_squared}")
    if plot:
        plt.figure(figsize=(10, 6))
        plt.semilogx(ts, g2)
        plt.semilogx(ts_fit, fit_curve)
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$g^{(2)}$')
        plt.title(r'Curve Fitting')
    return params_fit


###############################################
# two-time matrix method (direct)
###############################################    

def cal_g2mat(waterfall, method='vector'):
    '''
    Calculate two-time matrix directly from waterfall. May overflow memory if waterfall is large
    --------
    Parameters:
    waterfall : 3D numpy array, n_img * n_x * n_y
    --------
    Returns:
    g2mat : 2D numpy array, two-time correlation matrix
    '''
    if len(waterfall.shape) == 3:
        wf = waterfall.reshape(waterfall.shape[0], waterfall.shape[1]*waterfall.shape[2])
    if len(waterfall.shape) == 1:
        wf = waterfall.reshape(waterfall.shape[0], 1)
    
    if method == 'legacy':
        ab = np.zeros([waterfall.shape[0], waterfall.shape[0]])
        aa = np.zeros(waterfall.shape[0])
        for i in range(waterfall.shape[0]):
            for j in range(i):
                ab[i, j] = np.mean(np.multiply(waterfall[i], waterfall[j]))
                ab[j, i] = ab[i, j]
            ab[i, i] = np.mean(np.multiply(waterfall[i], waterfall[i]))
            aa[i] = np.mean(waterfall[i])
        return np.divide(ab, np.outer(aa, aa))

    if method == 'vector':
        wf_mean = wf.mean(axis=1).reshape(-1, 1)
        cov = (wf @ wf.T) / wf.shape[1]
        g2mat = cov / (wf_mean @ wf_mean.T)
        
    if method == 'n_cc':
        wf_squ_mean = np.average(wf**2, axis=1).reshape(-1, 1)
        g2mat = (wf @ wf.T) / wf.shape[1]
    return g2mat 
    

def gen_g2_from_g2mat(g2mat, mask=[]):
    ''' 
    Generate g2 from correlation matrix.
    --------
    Parameters:
    g2mat : 2d numpy array
    mask : 2d numpy array, same dim as g2mat
    --------
    Returns:
    g2: 1D numpy array, autocorrelation function
    '''
    if mask:
        g2mat = np.multiply(g2mat, mask)
    diag = [np.diagonal(g2mat, i) for i in range(g2mat.shape[0])]
    g2 = [np.nanmean(d) for d in diag]
    sigma_g2 = [np.nanstd(d) for d in diag]
    return np.array(g2), np.array(sigma_g2)

