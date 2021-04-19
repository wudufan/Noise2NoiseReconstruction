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
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[5]:


import argparse
parser = argparse.ArgumentParser(description = 'fbp reconstruction all sinograms')

# paths
parser.add_argument('--outDir', dest='outDir', type=str, default=None)
parser.add_argument('--sino', dest='sino', type=str, 
                    default='/home/dwu/DeepRecon/train/data/real_2D_3_layer_mean/L067_full_sino.npy')
parser.add_argument('--paramFile', dest='paramFile', type=str, 
                    default='/home/dwu/DeepRecon/train/data/param_real.txt')
parser.add_argument('--resolution', dest='resolution', type=float, default=0.8)

# simulation
parser.add_argument('--N0', dest='N0', type=float, default=-1)
parser.add_argument('--doseRate', dest='doseRate', type=float, default=0.25)
parser.add_argument('--filter', dest='filter', type=int, default=2, 
                    help='filter for fbp: 0-RL, 2-Hann')

# general network training
parser.add_argument('--device', dest='device', type=int, default=0)
parser.add_argument('--slices', dest='slices', type=int, nargs=2, default=[49, 50])
parser.add_argument('--outputInterval', dest='outputInterval', type=int, default=10)

# data augmentation
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[6]:


if sys.argv[0] != 'fbp_2d_from_origin.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
                              '--slices', '0', '50',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--outDir', '/home/dwu/DeepRecon/train/data/real_2D_3_layer_mean_resolution_0.8/'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[8]:


reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)
reconNet.cSetDevice(args.device)

reconNet.dx = args.resolution
reconNet.dy = args.resolution

if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)


# In[9]:


# read projections
prjs = np.load(args.sino)

if args.slices[0] >= prjs.shape[0]:
    args.slices[0] = prjs.shape[0] - 1
if args.slices[1] > prjs.shape[0]:
    args.slices[1] = prjs.shape[0]

prjs = np.copy(prjs[args.slices[0]:args.slices[1], ...], 'C') / args.imgNorm

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


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


# In[13]:


if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[15]:


if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
np.save(os.path.join(args.outDir, os.path.basename(args.sino).replace('sino', 'hann')), imgs * args.imgNorm)


# In[ ]:




