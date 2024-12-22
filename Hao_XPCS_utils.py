#!/usr/bin/python3

import numpy as np
import os, sys
import multiprocessing
import time
import itertools
import h5py
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm


##############################################
# Parameters needed
# indicate them at the beginning
##############################################

DATA_PATH = '/cds/home/h/haozheng/lw68/hdf5/smalldata/'
ROI_BOX = [200, 210, 500, 510]
SAVE_PATH = '/cds/home/h/haozheng/'
PREFIX='xpplx5019_Run'
KEY='jungfrau1M/ROI_0_area'



###############################################
# misc helper functions
###############################################

def pr_keys(d, level=0):
    try:
        k_lst = d.keys()
        for k in k_lst:
            print("\t"*level+f"-{k}")
            pr_keys(d[k], level=level+1)
    except:
        pass

      
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
    return img[box[2]:box[3], box[0]:box[1]]


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
    n_imgs, n_detect, n_x, n_y = imgs.shape
    
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
# two-time diagonal method
###############################################

@timeit
def cal_tt(waterfall, save=False, file=''):
    '''
    Calculate two-time diagonals from waterfall.
    --------
    Parameters:
    waterfall : 3D numpy array, n_img * n_pixels
    save : boolean, save tt to file, optional
    file : str, destination of saving
    --------
    Returns:
    tt : list of 1D numpy array, two-time diagonals
    '''
    norm = np.nanmean(waterfall, axis=1)
    tmpfall = waterfall/norm[:, None]
    tt = []
    for dt in range(0, waterfall.shape[0]):
        if dt:
            tmp = tmpfall[dt:]*tmpfall[:-dt]
            tt_dt = np.nanmean(tmp, axis=1)
        else:
            # dt = 0
            tmp = tmpfall*tmpfall
            tt_dt = np.nanmean(tmp, axis=1)
        tt.append(tt_dt)
        if save:
            if not f_name:
                f_name = '/cds/home/h/haozheng/twotime/Untitled.h5'
            h5f.create_dataset(f'{i}', data=tt[i])
    return tt


def save_tt_to_h5(file, tt):
    '''
    Save two-time diagonals to h5 file
    --------
    Parameters:
    file : str, destination
    tt : list of 1D numpy array, two-time diagonals
    --------
    Returns:
    '''
    h5f = h5py.File(file, 'w')
    for i in range(len(tt)):
        h5f.create_dataset(f'{i}', data=tt[i])
    h5f.close()


def read_tt_from_h5(file):
    '''
    Read tt from saved h5 file
    --------
    Parameters:
    file : str, source
    --------
    Returns:
    tt : list of 1D numpy array, two-time diagonals
    '''
    h5f = h5py.File(file, 'r')
    tt = [h5f[f'{i}'][:] for i in range(len(h5f))]
    h5f.close()
    return tt


def gen_g2mat_from_tt(tt):
    '''
    Generate two-time matrix from two-time diagonals
    --------
    Parameters:
    tt : list of 1D numpy array, two-time diagonals
    --------
    Returns:
    g2mat : 2D numpy array, two-time correlation matrix
    '''
    num_steps = len(tt)
    g2mat = np.zeros((num_steps, num_steps))
    for i in range(num_steps):
        for j in range(num_steps-i):
            g2mat[j, i+j] = tt[i][j]
            g2mat[i+j, j] = tt[i][j]
    return g2mat


@timeit
def gen_g2_from_tt(tt):
    '''
    Generate autocorrelation from two-time diagonals
    --------
    Parameters:
    tt : list of 1D numpy array, two-time diagonals
    --------
    Returns:
    g2: 1D numpy array, autocorrelation function
    '''
    return np.array([np.mean(dt) for dt in tt])


def cal_tt_helper(tmpfall, dts):
    '''
    Subroutine for calculating two-time diagonals
    --------
    Parameters:
    tmpfall : 2D numpy array, normalized waterfall images
    dts : list, list of tau
    --------
    Returns:
    tt : list of 1D numpy array, two-time diagonals
    '''
    tt = []
    for dt in dts:
        if dt:
            tmp = tmpfall[dt:]*tmpfall[:-dt]
            tt_dt = np.nanmean(tmp, axis=1)
        else:
            # dt = 0
            tmp = tmpfall*tmpfall
            tt_dt = np.nanmean(tmp, axis=1)
        tt.append(tt_dt)
    return tt


@timeit    
def cal_tt_parallelized(waterfall, n_proc=8, save='False', f_name=''):
    '''
    Calculate two-time diagonal using multiple threads
    --------
    Parameters:
    waterfall : 3D numpy array, n_img * n_pixels
    n_proc : int, number of threads
    save : boolean, flag for saving tt
    f_name : str, destination file
    --------
    Returns:
    tt : list of 1D numpy array, two-time diagonals
    '''
    n_frames = waterfall.shape[0]
    norm = np.nanmean(waterfall, axis=1)
    tmpfall = waterfall/norm[:, None]
    dts_array = n_in_m_bins(n_frames, n_proc)
    with multiprocessing.Pool(processes=n_proc) as pool:
        proc_results = [pool.apply_async(cal_tt_helper,
                                         args=(tmpfall, dts)) for dts in dts_array]
        tt = list(itertools.chain(*[r.get() for r in proc_results]))
        tt.sort(key=len, reverse=True)
    if save:
        if not f_name:
            f_name = '/cds/home/h/haozheng/twotime/Untitled.h5'
        save_tt_to_h5(f_name, tt)
    return tt


###############################################
# g2 curve fitting
###############################################

def siegert(t, beta, tau):
    '''
    Siegert function for fitting autocorrelation vs tau
    g2 = 1 + beta * exp(- 2 * tau/ tau_c)
    params: beta, tau_c
    '''
    return 1 + beta * np.exp(-2 * t / tau)


def fit_g2(g2, ts=[], fit_range='full', func=siegert, params_init=(1, 30), plot=True):
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
        plt.figure(figsize=(16, 9))
        plt.semilogx(ts, g2)
        plt.semilogx(ts_fit, fit_curve)
        plt.xlabel(r'Delay $\tau$')
        plt.ylabel(r'Autocorrelation Function $g^{(2)}$')
        plt.title(r'Curve Fitting')


###############################################
# two-time matrix method (direct)
###############################################    

def cal_g2mat(waterfall):
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
        waterfall = waterfall.reshape(waterfall.shape[0], waterfall.shape[1]*waterfall.shape[2])
    if len(waterfall.shape) == 1:
        waterfall = waterfall.reshape(waterfall.shape[0], 1)
    
    ab = np.zeros([waterfall.shape[0], waterfall.shape[0]])
    aa = np.zeros(waterfall.shape[0])
    for i in range(waterfall.shape[0]):
        for j in range(i):
            ab[i, j] = np.mean(np.multiply(waterfall[i], waterfall[j]))
            ab[j, i] = ab[i, j]
        ab[i, i] = np.mean(np.multiply(waterfall[i], waterfall[i]))
        aa[i] = np.mean(waterfall[i])
    return np.divide(ab, np.outer(aa, aa))


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


###############################################
# two-time matrix (overlap blocks)
###############################################    
    

def save_g2mat_blocks(g2_blocks, run_num, roi_fn, roi_params, ini_imgs, fin_imgs):
    dir_name =  f'g2mat_blocks_Run{run_num}_{roi_fn.__name__}{roi_params}'
    save_path = os.path.join(SAVE_PATH, dir_name)
    if not os.path.isdir(save_path):
        os.mkdir(os.path.join(save_path))
    for block, ini, fin in zip(g2_blocks, ini_imgs, fin_imgs):
        f_name = f'g2_block_[{ini}, {fin}].csv'
        np.savetxt(os.path.join(save_path, f_name), block)

        
def plot_g2mat_blocks(g2_blocks, xy_mesh):
    plt.figure(figsize=(16, 9))
    for g2_block, [x_mesh, y_mesh] in zip(g2_blocks, xy_mesh):
        plt.pcolormesh(x_mesh, y_mesh, g2_block, vmin=1, vmax=2, cmap='jet', shading='auto')
    plt.gca().set_aspect(1)
    

@timeit
def cal_g2_from_tiles(waterfall, ini_imgs, fin_imgs):
    '''
    Calculate g2 from tiles of two-time matrix blocks
    --------
    Parameters:
    waterfall : 3D numpy array, n_img * n_x * n_y
    ini_imgs : list, list of indices of initial images
    fin_imgs : list, list of indices of final images
    --------
    Returns:
    g2_blocks : list, list of g2mat blocks
    '''
    return [cal_g2mat(waterfall[ini:fin]) for ini, fin in zip(ini_imgs, fin_imgs)]


@timeit
def cal_g2_blocks_parallelized(waterfall, ini_imgs, fin_imgs, n_proc=8):
    '''
    Calculate g2 from tiles of two-time matrix blocks, multi-thread version
    --------
    Parameters:
    waterfall : 3D numpy array, n_img * n_x * n_y
    ini_imgs : list, list of indices of initial images
    fin_imgs : list, list of indices of final images
    n_proc : int, number of threads
    --------
    Returns:
    g2_blocks : list, list of g2mat blocks
    '''
    with multiprocessing.Pool(processes=n_proc) as pool:
        proc_results = [pool.apply_async(cal_g2mat,
                                         args=(waterfall[ini:fin],)) for ini, fin in zip(ini_imgs, fin_imgs)]
        g2_blocks = [r.get() for r in proc_results]
    return g2_blocks


def speckle_ana_block(run_num,
                prefix='xpplx5019_Run',  
                key='jungfrau1M/ROI_0_area',
                roi_fn=box_roi,
                roi_params=[],
                block_size=1000, 
                overlap=500, 
                n_proc=8, 
                plot=True, 
                save=True):
    waterfall, stats = h5_stats(run_num=run_num, prefix=prefix, key=key,
                                roi_fn=box_roi,roi_params=roi_params,save=save)
    
    n_imgs = waterfall.shape[0]
    ini_imgs = np.arange(0, n_imgs, block_size-overlap)
    fin_imgs = np.arange(block_size, n_imgs, block_size-overlap)
    ini_imgs = ini_imgs[:len(fin_imgs)]
    
    print('Calculating g2mat blocks...')
    g2_blocks = cal_g2_blocks_parallelized(waterfall, ini_imgs, fin_imgs, n_proc=n_proc)
    if save:
        save_g2mat_blocks(g2_blocks, run_num, roi_fn, roi_params, ini_imgs, fin_imgs)
    if plot:
        xy_mesh = [np.meshgrid(np.arange(ini, ini+block_size), 
                               np.arange(ini, ini+block_size)) for ini, fin in zip(ini_imgs, fin_imgs)]
        plot_g2mat_blocks(g2_blocks, xy_mesh)
    return gen_g2_from_g2mat(np.mean(g2_blocks, axis=0))


###############################################
# frame-frame correlation sequence
###############################################

def cal_frame_corr(a, b):
    '''
    Calculate correlation between 2 frames
    --------
    Parameters
    a : 1D numpy array, flattened frame
    a : 1D numpy array, flattened frame, same dim as a
    '''
    # return np.mean(np.multiply(a, b)) / np.mean(np.outer(a, a))
    return np.mean(np.multiply(a, b)) / (np.mean(a) * np.mean(b))


def cal_frame_corr_seq(waterfall, flags=[]):
    n_imgs = waterfall.shape[0]
    frame_corr_seq_f1 = []
    frame_corr_seq_f0 = []
    if flags:
        for i in range(1, n_imgs):
            corr = cal_frame_corr(waterfall[i-1], waterfall[i])
        if flags[i] == 1:
            frame_corr_seq_f1.append(corr)
        else:
            frame_corr_seq_f0.append(corr)
    else:
        for i in range(1, n_imgs):
            frame_corr_seq_f0.append(cal_frame_corr(waterfall[i-1], waterfall[i]))
    return np.array(frame_corr_seq_f0), np.array(frame_corr_seq_f1)


def save_frame_corr_seq(frame_corr_seq, run_num, roi_fn, roi_params, flag=0):
    f_name =  f'f{flag}_corr_seq_Run{run_num}_{roi_fn.__name__}{roi_params}.csv'
    np.savetxt(os.path.join(SAVE_PATH, f_name), frame_corr_seq)


def speckle_ana_seq(run_num,
                prefix='xpplx5019_Run',  
                key='jungfrau1M/ROI_0_area',
                roi_fn=box_roi,
                roi_params=[],
                n_proc=8, 
                flags=[],
                plot=False, 
                save=True):
    waterfall, stats = h5_stats(run_num=run_num, prefix=prefix, key=key,
                                roi_fn=box_roi,roi_params=roi_params,save=save)
    print('Calculating frame_corr_seq...')
    frame_corr_seq_f0, frame_corr_seq_f1 = cal_frame_corr_seq(waterfall, flags=flags)
    if save:
        save_frame_corr_seq(frame_corr_seq_f0, run_num, roi_fn, roi_params, flag=0)
        save_frame_corr_seq(frame_corr_seq_f1, run_num, roi_fn, roi_params, flag=1)
    return frame_corr_seq_f0, frame_corr_seq_f1


if __name__ == '__main__':
    # method using g2mat 
    g2, g2_sigma = speckle_ana_block(run_num=145, 
                                 prefix=PREFIX, 
                                 key=KEY, 
                                 roi_fn=box_roi, 
                                 roi_params=ROI_BOX,
                                 block_size=1000, 
                                 overlap=0, 
                                 n_proc=8, 
                                 plot=False, 
                                 save=True)
    fit_g2(g2, ts=[], fit_range='full', func=siegert, params_init=(1, 30), plot=False)


    # method using only frame-frame sequence
    frame_corr_seq_f0, frame_corr_seq_f1 = speckle_ana_seq(run_num=145,
                                                        prefix='xpplx5019_Run',  
                                                        key='jungfrau1M/ROI_0_area',
                                                        roi_fn=box_roi,
                                                        roi_params=ROI_BOX,
                                                        n_proc=8,  
                                                        save=True)
