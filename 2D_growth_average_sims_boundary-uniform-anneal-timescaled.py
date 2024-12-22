import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm, patches, animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec
from matplotlib.colors import LogNorm

# For papers - 
# All figs will be 14 inch for two columns and 7 in for one column


# importing system packages
import os
import sys
import glob
import h5py
import time
import itertools

from tqdm import tqdm

# importing the workhorse
import numpy as np
from numpy.random import default_rng
rng = default_rng()
import pandas as pd
from scipy import io, signal, interpolate, ndimage
from scipy.special import gamma, erf
from skimage import filters
# import seaborn as sns

from sg2d import *

# tiff packages
import tifffile

from lmfit import Minimizer, Parameters, report_fit

from XPCSana import *
from visReduce import *

# from Hao_speckle_utils import *
class nucleus:
    def __init__(self, x_center,y_center,init_rad,index=0,soft_edge=0,growth_rate=1):
        self.xCen=x_center
        self.yCen=y_center
        self.rad=init_rad
        self.soft_edge=soft_edge
        self.growth_rate=growth_rate
        self.index=index

# utils
def genInitNuclei(nSize, xCen, yCen, rCen,soft_edges=None,growth_rates=None):
    x = np.arange(nSize)
    y= np.arange(nSize)
    xx, yy = np.meshgrid(x, y)

    nuclei=[]
    se=soft_edges is None
    gr=growth_rates is None
    
    if se and gr:
        for ind in range(len(rCen)):
            nuclei.append(nucleus(xCen[ind],yCen[ind],rCen[ind],ind))
    elif se and not gr:
        for ind in range(len(rCen)):
            nuclei.append(nucleus(xCen[ind],yCen[ind],rCen[ind],ind,growth_rate=growth_rates[ind]))
    elif not se and gr:
        for ind in range(len(rCen)):
            nuclei.append(nucleus(xCen[ind],yCen[ind],rCen[ind],ind,soft_edge=soft_edges[ind]))
    elif not se and not gr:
        for ind in range(len(rCen)):
            nuclei.append(nucleus(xCen[ind],yCen[ind],rCen[ind],ind,soft_edge=soft_edges[ind],growth_rate=growth_rates[ind]))
            
    return xx,yy,nuclei

def genImgs(nuclei,xx,yy,boundaries=None,soften_edges=None):
    img=np.zeros_like(xx,dtype=float)
    if boundaries is None:
        boundaries=np.zeros_like(img)
    for i in range(len(nuclei)):
        distSq = (xx-nuclei[i].xCen)**2+(yy-nuclei[i].yCen)**2
        img=np.logical_or(distSq<nuclei[i].rad**2,img).astype(float)
        
#     img = img.astype(np.float64) * boundaries.astype(np.float64)
    if soften_edges=='blur':
        img=img.astype(np.float64)
        img=ndimage.gaussian_filter(img,3)
    elif soften_edges=='individual' or soften_edges=='random':
        for i in range(len(nuclei)):
            distSq = (xx-nuclei[i].xCen)**2+(yy-nuclei[i].yCen)**2
            mask= np.logical_and(distSq>=nuclei[i].rad**2,distSq<=(nuclei[i].rad+nuclei[i].soft_edge)**2)
            img[mask]=img[mask]+0.5
        too_large=np.where(img>1.0)
        img[too_large]=1.0
    bnd_row,bnd_col=np.nonzero(boundaries)
    for row,col in zip(bnd_row,bnd_col):
        if img[row,col]>0.5:
            if soften_edges is None:
                img[row,col]=0
            else:
                img[row,col]=0.5
    
    if soften_edges=='random':
        bnd_row,bnd_col=np.nonzero(img==0.5)
        for row,col in zip(bnd_row,bnd_col):
            img[row,col]=np.random.rand()
    return img

def genSpeckle(imgStk):
    # Generate speckles
    speckle = np.zeros(imgStk.shape)
    
    dim = len(imgStk.shape)

    if dim==2:
        temp = np.fft.fftshift(np.fft.fft2(imgStk))
        speckle = np.absolute(temp)**2
    elif dim==3:
        temp = np.fft.fftshift(np.fft.fft2(imgStk),axes=(-2,-1))
        speckle = np.absolute(temp)**2

#     speckle = speckle.astype(np.int32)
    
    return speckle

def growNuclei(nuclei,newGR=None):
    if newGR is None:
        for nuc in nuclei:
            nuc.rad+=nuc.growth_rate
    else:
        for ind,nuc in enumerate(nuclei):
            nuc.rad+=nuc.growth_rate
            nuc.growth_rate=newGR[ind]
        
def produceNucCoords(nuclei):
    xs=[]
    ys=[]
    for nuc in nuclei:
        xs.append(nuc.xCen)
        ys.append(nuc.yCen)
    return np.array(xs),np.array(ys)

def calculateBoundariesRatios(nuclei,nSize):
    xx,yy=np.meshgrid(np.arange(nSize).astype(np.float64),np.arange(nSize).astype(np.float64))
    ratios=np.ones((nSize,nSize),dtype=np.float64)
    pair_index=ratios.copy()
    d1=ratios.copy()*np.inf
    d2=ratios.copy()*np.inf
    n1=np.zeros_like(d1)
    n2=np.zeros_like(d2)
    old_d=np.ones_like(ratios)*np.inf
    old_n=-1
    for ind,nuc in enumerate(nuclei):
        d=(xx-nuc.xCen)**2+(yy-nuc.yCen)**2
        d1_inds=d<d1
        d2_inds=np.logical_and(d>=d1,d<=d2)
        d1[d1_inds]=d[d1_inds]
        n1[d1_inds]=nuc.index
        d2[d1_inds]=old_d[d1_inds]
        n2[d1_inds]=old_n
        d2[d2_inds]=d[d2_inds]
        n2[d2_inds]=nuc.index
        old_d=d.copy()
        old_n=nuc.index
#         fig,axis=plt.subplots(1,2)
#         axis[0].imshow(d1,origin='lower')
#         axis[1].imshow(d2,origin='lower')
    ratios=(d2-d1)/(d2+d1)
    pair_index=((n1+n2)+np.abs(n1-n2))*((n1+n2)+np.abs(n1-n2)+1)/2+(n1+n2)
#     plt.figure()
#     plt.imshow(ratios,origin='lower')
    
    return ratios,pair_index

def long_tail_sampling_conversion(uniforms,alpha,tau):
    return tau*((-uniforms+1)**(-1/alpha)-1)

# Defining the parameters
nCen = 1000 # Number of nucleus
nSize = 500 # Field of view
nSteps_growth = 51 # Steps
grow_time=1000
nSteps_anneal= 51
anneal_time=1e9
timeSteps=np.concatenate((1+np.floor(np.linspace(0,grow_time,nSteps_growth)),1+grow_time+np.logspace(2,np.log10(anneal_time),nSteps_anneal)))
nChange = 4 # Number of nucleus changed
iniR = 4 # Initial max nucleaus size
rate = np.sqrt((nSize**2)/nCen)/grow_time # Growth rate
newCen = 0 # New centers
boundary_thres_start=0.05
boundary_thres_final=0.01
# anneal_alpha=3/4  #for comparison to exponential distribution, use this line if anneal_tau's value depends on anneal_alpha
anneal_end_time=2e6
ROI_mins=np.array([2,12])
ROI_maxs=np.array([10,15])
num_iters=250
corr_funcs=np.zeros((num_iters,nSteps_growth+nSteps_anneal+1,len(ROI_mins)))
intensity_funcs=np.zeros((num_iters,nSteps_growth+nSteps_anneal+1))
new_phase_area=np.zeros_like(intensity_funcs)
percents_anneal=np.zeros((num_iters,nSteps_anneal))

# foldername='cell auto results/{}nucs_{}size_{}steps_{}rate-uni-const_rand-ave-100'.format(nCen,nSize,nSteps,rate)
foldername='cell auto results/log-time-early-test/include-growth/scaled-t/starting-config-comp/{}nucs_{}size_{}growth-scaled-time-{:.2E}rate-markov-poisson_{}anneal_hard/endtime{:.1E}-uniform-anneal_bound{:.1E}-{:.1E}-ave{}'.format(nCen,nSize,nSteps_growth,rate,nSteps_anneal,anneal_end_time,boundary_thres_start,boundary_thres_final,num_iters)
if not os.path.isdir(foldername):
    os.makedirs(foldername)
print('folder', foldername)

# xCen = np.random.randint(0, high=nSize, size=nCen)
# yCen = np.random.randint(0, high=nSize, size=nCen)

# ravelCen=rng.choice(np.arange(nSize**2),size=nCen,replace=False)
# xCen,yCen=np.unravel_index(ravelCen,shape=(nSize,nSize))

# rCen = rng.integers(3, high=iniR+1, size=nCen)

starting_dict=np.load('starting_nuc_pos_rad.npz')
xCen,yCen,rCen=starting_dict['xCen'],starting_dict['yCen'],starting_dict['rCen']

num_nuclei_grow=rng.poisson(rate**-1,grow_time+1)
grow_rates=np.zeros_like(rCen)
grow_rates[rng.choice(np.arange(nCen),size=num_nuclei_grow[0],replace=False)]=1
xx,yy,nuclei=genInitNuclei(nSize, xCen, yCen, rCen,growth_rates=grow_rates)

pixel_boundary_ratios,pair_index=calculateBoundariesRatios(nuclei,nSize)
boundaries=pixel_boundary_ratios<boundary_thres_start
final_bndys=pixel_boundary_ratios<boundary_thres_final
    # boundaries=None
    # Start evolution
for rep in tqdm(range(num_iters)):
    imgStk = []
    if rep>0:
        num_nuclei_grow=rng.poisson(rate**-1,grow_time+1)
        grow_rates=np.zeros_like(rCen)
        grow_rates[rng.choice(np.arange(nCen),size=num_nuclei_grow[0],replace=False)]=1
        xx,yy,nuclei=genInitNuclei(nSize, xCen, yCen, rCen,growth_rates=grow_rates)
    img = genImgs(nuclei, xx,yy)
    imgStk.append(img)
    prev_time=0
    for j in range(nSteps_growth):
        for timeStep in np.arange(prev_time,timeSteps[j]):
            new_grow_rates=np.zeros_like(rCen)
            new_grow_rates[rng.choice(np.arange(nCen),size=num_nuclei_grow[j],replace=False)]=1
         # Growth
        # new_grow_rates=rng.normal(rate,rate/1.2,size=rCen.shape)
    #     new_grow_rates=np.random.randint(0,rate+1,size=rCen.shape)+0.5
    #     new_grow_rates=np.random.choice([0,rate],size=rCen.shape)
    #     new_grow_rates=None
            growNuclei(nuclei,newGR=new_grow_rates)
        prev_time=timeSteps[j]
        # print(prev_time)
        # # Nucleation
        # xCen = np.hstack((xCen, np.random.randint(0, high=nSize, size=newCen)))
        # yCen = np.hstack((yCen, np.random.randint(0, high=nSize, size=newCen)))
        # rCen = np.hstack((rCen, np.random.randint(1, high=iniR+1, size=newCen)))

        img = genImgs(nuclei,xx,yy,boundaries,soften_edges=None)
        imgStk.append(img)

    
    border_inds=np.unique(pair_index)
    border_anneal_times=rng.integers(grow_time,anneal_end_time,border_inds.shape)
    # vals,bins,art=plt.hist(border_anneal_times,100)
    # cdf=np.array([np.sum(border_anneal_times<(k+1)) for k in range(nSteps_anneal)])
    # plt.plot(cdf)
    for step in range(nSteps_anneal):
        timeStep=timeSteps[nSteps_growth+step]
        convert_pair_inds=border_inds[np.where(np.logical_and(border_anneal_times>prev_time,border_anneal_times<=timeStep))]
        percents_anneal[rep,step]=convert_pair_inds.shape[0]/border_inds.shape[0]
        img=imgStk[-1].copy()
        for ind in convert_pair_inds:
            change_region=np.where(pair_index==ind)
            img[change_region]=np.logical_not(final_bndys[change_region])
        imgStk.append(img)
        prev_time=timeStep

    imgStk=np.array(imgStk)
    xpcsStk=genSpeckle(imgStk)
    intensity_funcs[rep,:]=np.sum(xpcsStk,axis=(1,2))/np.sum(xpcsStk[-1])
    new_phase_area[rep,:]=np.sum(imgStk,axis=(1,2))
    # if rep==0:
    #     tifffile.imwrite(os.path.join(foldername,'imgStk0.tif'),imgStk)
    #     tifffile.imwrite(os.path.join(foldername,'xpcsStk0.tif'),xpcsStk)
    for ind,ROI_min in enumerate(ROI_mins):
        ROI_max=ROI_maxs[ind]
        waterfall = genCirROI(xpcsStk, ROI_min, ROI_max)
        for row in range(waterfall.shape[0]):
            waterfall[row,:]=waterfall[row,:]/np.mean(waterfall[row,:])
        tt=calcTwoTime(waterfall,timing=False)
        corr_funcs[rep,:,ind]=tt[-1,:]
        


num_prev_runs=len(glob.glob(os.path.join(foldername,'multi_average_arrays*.npz')))
np.savez(os.path.join(foldername,'multi_average_arrays_w-area{}.npz'.format(num_prev_runs)),corr_funcs=corr_funcs,intensity_funcs=intensity_funcs,ROI_mins=ROI_mins,ROI_maxs=ROI_maxs,new_phase_area=new_phase_area,timeSteps=timeSteps,percents_anneal=percents_anneal)