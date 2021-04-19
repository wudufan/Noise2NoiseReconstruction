#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import scipy
import sporco
from sporco import util
from sporco.dictlrn import cbpdndl

sys_pipes = sporco.util.notebook_system_output()


# In[2]:


sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'csc reconstruction fp pre training of dictionaries')

# paths
parser.add_argument('--checkPoint', dest='checkPoint', type=str, default=None)
parser.add_argument('--sino', dest='sino', type=str, 
                    default='../../train/recon/data/sino/quarter_sino.npy')
parser.add_argument('--ref', dest='ref', type=str, 
                    default='../../train/recon/data/sino/full_gaussian.npy')
parser.add_argument('--paramFile', dest='paramFile', type=str, 
                    default='../../train/recon/data/sino/param.txt')
parser.add_argument('--resolution', dest='resolution', type=str, 
                    default='../../train/recon/data/sino/resolutions.npy')

# simulation
parser.add_argument('--N0', dest='N0', type=float, default=-1)
parser.add_argument('--filter', dest='filter', type=int, default=2, 
                    help='filter for fbp: 0-RL, 2-Hann')
parser.add_argument('--doseRate', dest='doseRate', type=float, default=0.25)

# general network training
parser.add_argument('--device', dest='device', type=int, default=0)
parser.add_argument('--nSlices', dest='nSlices', type=int, default=10)
parser.add_argument('--outputInterval', dest='outputInterval', type=int, default=50)

# csc
parser.add_argument('--kernelSize', dest='kernelSize', type=int, nargs=2, default = [10,10])
parser.add_argument('--nChannels', dest='nChannels', type=int, default = 32)
parser.add_argument('--lmbda', dest='lmbda', type=float, default = 0.005)
parser.add_argument('--nPreIter', dest='nPreIter', type=int, default = 1000)
# prefilter for csc
parser.add_argument('--lmbdaPre', dest='lmbdaPre', type=float, default = 1)

# data augmentation
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[4]:


if sys.argv[0] != 'pretrain_2d.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
#                               '--N0', '1e5',
                              '--nSlices', '10', 
                              '--nPreIter', '10',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--checkPoint', '../../train/recon/csc_2d/pretrain/kernel_10_channel_32_lmbda_0.005'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[5]:


reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)
reconNet.cSetDevice(args.device)


# In[6]:


# read phantom
np.random.seed(0)
x = np.load(args.ref)
resolutions = np.load(args.resolution)

inds = np.arange(x.shape[0])
np.random.shuffle(inds)
x = x[inds[:args.nSlices], ...]
resolutions = resolutions[inds[:args.nSlices]]

refs = x / args.imgNorm

if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[7]:


# read projections
prjs = np.load(args.sino)
prjs = np.copy(prjs[inds[:args.nSlices], ...], 'C') / args.imgNorm

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


# In[8]:


# mask for network training
masks = helper.GetMasks2D(reconNet, resolutions)


# In[9]:


# fbp
imgs = []
for i in range(prjs.shape[0]):
# for i in range(10):
    if i % 10 == 0:
        print (i, end=', ')
    
    reconNet.dx = resolutions[i]
    reconNet.dy = resolutions[i]
    
    fp = prjs[[i],...]    
    
    fsino = reconNet.cFilter3d(np.copy(fp[[0], ...], 'C'), args.filter)
    imgs.append(reconNet.cDDFanBackprojection3d(np.copy(fsino, 'C'), type_projector=1))
imgs = np.concatenate(imgs, 0)


# In[10]:


if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[20]:


# training, offline learning
np.random.seed(0)

s = np.transpose(imgs[..., 0], (1, 2, 0))
sl, sh = util.tikhonov_filter(s, args.lmbdaPre)

D0 = np.random.normal(size = args.kernelSize + [args.nChannels]).astype(np.float32)

opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': args.nPreIter,
                                         'CBPDN': {'rho': 100*args.lmbda + 1, 
                                                   'AutoRho': {'Enabled': True}, 
                                                   'RelaxParam': 1.8, 
                                                   'RelStopTol': 1e-7},
                                         'CCMOD': {'rho': 10.0, 'ZeroMean': False}}, 
                                        dmethod='cns')
learner = cbpdndl.ConvBPDNDictLearn(D0, sh, args.lmbda, opt, dmethod='cns')
learner.solve()

D1 = learner.getdict()
D1 = D1.squeeze()

if not os.path.exists(os.path.dirname(args.checkPoint)):
    os.makedirs(os.path.dirname(args.checkPoint))
np.save(args.checkPoint, D1)


# In[21]:


if showPlots:
    plt.imshow(sporco.util.tiledict(D1))


# In[ ]:




