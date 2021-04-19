#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import scipy


# In[2]:


sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
import PriorFunctionSolver
sys.path.append('/home/dwu/DeepRecon/DECT/HYPR_NLM/python/')
import HYPR_NLM
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[4]:


import argparse
parser = argparse.ArgumentParser(description = 'fbp reconstruction sinogram')

# paths
parser.add_argument('--outDir', dest='outDir', type=str, default=None)
parser.add_argument('--sino', dest='sino', type=str, 
                    default='../../train/recon/data/sino/quarter_sino.npy')
parser.add_argument('--ref', dest='ref', type=str, 
                    default='../../train/recon/data/sino/full_gaussian.npy')
parser.add_argument('--paramFile', dest='paramFile', type=str, 
                    default='../../train/recon/data/sino/param.txt')
parser.add_argument('--resolutionFile', dest='resolutionFile', type=str, 
                    default='../../train/recon/data/sino/resolutions.npy')

# simulation
parser.add_argument('--N0', dest='N0', type=float, default=-1)
parser.add_argument('--doseRate', dest='doseRate', type=float, default=0.25)
parser.add_argument('--filter', dest='filter', type=int, default=2, 
                    help='filter for fbp: 0-RL, 2-Hann')

# general network training
parser.add_argument('--device', dest='device', type=int, default=0)
parser.add_argument('--slices', dest='slices', type=int, nargs=2, default=[68, 69])
parser.add_argument('--outputInterval', dest='outputInterval', type=int, default=10)

# data augmentation
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[13]:


if sys.argv[0] != 'fbp_2d.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
                              '--sino', '../../train/recon/data/sino/full_sino.npy',
#                               '--ref', '../../train/recon/data/fp/full_hann.npy',
                              '--slices', '0', '100',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--outDir', '../../train/recon/data/fp'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[6]:


reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)
reconNet.cSetDevice(args.device)

if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)


# In[7]:


# read phantom
x = np.load(args.ref)
if args.slices[0] >= x.shape[0]:
    args.slices[0] = x.shape[0] - 1
if args.slices[1] > x.shape[0]:
    args.slices[1] = x.shape[0]
x = np.copy(x[args.slices[0]:args.slices[1], ...], 'C')

# support only one resolution
resolution = np.load(args.resolutionFile)
resolution = np.mean(resolution[args.slices[0]:args.slices[1]])
reconNet.dx = resolution
reconNet.dy = resolution

refs = x / args.imgNorm

if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[8]:


# read projections
prjs = np.load(args.sino)
prjs = np.copy(prjs[args.slices[0]:args.slices[1], ...], 'C') / args.imgNorm

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


# In[9]:


# mask for network training
masks = helper.GetMasks2D(reconNet, [0.8] * refs.shape[0])


# In[10]:


# fbp
imgs = []
for i in range(prjs.shape[0]):
# for i in range(10):
    if i % 10 == 0:
        print (i, end=', ')
    
    fp = prjs[[i],...]    
    
    fsino = reconNet.cFilter3d(np.copy(fp[[0], ...], 'C'), args.filter)
    imgs.append(reconNet.cDDFanBackprojection3d(np.copy(fsino, 'C'), type_projector=1))
imgs = np.concatenate(imgs, 0)
x = imgs


# In[11]:


if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[14]:


if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
np.save(os.path.join(args.outDir, 'recon'), x * 0.019)
# np.savez(os.path.join(args.outDir, 'loss'), header = header, val = vals)
# with open(os.path.join(args.outDir, 'args'), 'w') as f:
#     for k in args.__dict__:
#         f.write('%s = %s\n'%(k, str(args.__dict__[k])))


# In[15]:


if showPlots:
    plt.figure(figsize=[18,12])
    plt.subplot(231); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(232); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(233); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(234); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(235); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(236); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)


# In[ ]:




