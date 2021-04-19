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


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'nlm reconstruction 2d')

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

# general iteration
parser.add_argument('--nIters', dest='nIters', type=int, default=100)
parser.add_argument('--nSubsets', dest='nSubsets', type=int, default=12)
parser.add_argument('--nesterov', dest='nesterov', type=float, default=0.5)
parser.add_argument('--betaRecon', dest='betaRecon', type=float, default = 0)
parser.add_argument('--eps', dest='eps', type=float, default = 1e-8)

# nlm
parser.add_argument('--searchSize', dest='searchSize', type=int, nargs=3, default=(1,11,11))
parser.add_argument('--kernelSize', dest='kernelSize', type=int, nargs=3, default=(1,5,5))
parser.add_argument('--kernelStd', dest='kernelStd', type=float, default=1)
parser.add_argument('--sigma', dest='sigma', type=float, default=0.05)
parser.add_argument('--fixedGuide', dest='fixedGuide', type=int, default=1)

# data augmentation
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[4]:


if sys.argv[0] != 'nlm_2d.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '1',
                              '--betaRecon', '5e-4',
                              '--sigma', '0.05',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--outDir', '../../train/recon/nlm_2d/test'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[5]:


reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)
reconNet.cSetDevice(args.device)

if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)


# In[6]:


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


# In[7]:


# read projections
prjs = np.load(args.sino)
prjs = np.copy(prjs[args.slices[0]:args.slices[1], ...], 'C') / args.imgNorm

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


# In[8]:


# mask for network training
masks = helper.GetMasks2D(reconNet, [resolution] * refs.shape[0])


# In[9]:


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


# In[10]:


if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[11]:


def CalcProjectorNorm(reconNet, weight, nIter = 20):
    weight = np.sqrt(weight)
    
    x = np.random.random_sample([1, reconNet.nx, reconNet.ny, 1]).astype(np.float32)
    x = x / np.linalg.norm(x)

    for i in range(nIter):
        print (i, end=',', flush=True)
        fp = reconNet.cDDFanProjection3d(x) * weight
        projectorNorm = np.linalg.norm(fp)
        x = reconNet.cDDFanBackprojection3d(fp * weight)

        x = x / np.linalg.norm(x)
    print ('')

    return projectorNorm


# In[12]:


weights = np.sqrt(np.exp(-prjs * args.imgNorm))
projectorNorm = CalcProjectorNorm(reconNet, weights)
normImg = reconNet.cDDFanNormImg3d(prjs, weights) / projectorNorm / projectorNorm
# masks = np.ones_like(masks)


# In[13]:


x = imgs
x0 = np.copy(x)
xd = HYPR_NLM.NLM(x, x0, args.searchSize, args.kernelSize, args.kernelStd, args.sigma)

if showPlots:
    plt.figure(figsize=[12,6])
    plt.subplot(121); plt.imshow(x0[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(122); plt.imshow(xd[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    rmse0_roi = np.sqrt(np.mean((x0 - refs)[0, 128:-128, 128:-128, 0]**2))
    print (rmse0_roi)


# In[14]:


def SQSOneStep(reconNet, x, z, ref, prj, weight, normImg, projectorNorm, args, verbose = 0):
    # projection term
    if not reconNet.rotview % args.nSubsets == 0:
        raise ValueError('reconNet.rotview cannot be divided by args.nSubsets')
    
    inds = helper.OrderedSubsetsBitReverse(reconNet.rotview, args.nSubsets)
    angles = np.array([reconNet.angles[i] for i in inds], np.float32)
    prj = prj[:, :, inds, :]
    weight = weight[:, :, inds, :]
    
    nAnglesPerSubset = int(reconNet.rotview / args.nSubsets)
    
    x_new = np.copy(z)
    
    for i in range(args.nSubsets):
        if verbose:
            print ('set%d'%i, end=',', flush=True)
        curAngles = angles[i*nAnglesPerSubset:(i+1)*nAnglesPerSubset]
        curWeight = weight[:, :, i*nAnglesPerSubset:(i+1)*nAnglesPerSubset, :]
        fp = reconNet.cDDFanProjection3d(x_new, curAngles) / projectorNorm
        dprj = fp - prj[:, :, i*nAnglesPerSubset:(i+1)*nAnglesPerSubset, :] / projectorNorm
        bp = reconNet.cDDFanBackprojection3d(curWeight * dprj, curAngles) / projectorNorm
        
#         tvs1, tvs2, _ = PriorFunctionSolver.cTVSQS2D(x_new, args.eps)
        sqsNlm = 4 * (x_new - HYPR_NLM.NLM(x_new, ref, 
                                           args.searchSize, 
                                           args.kernelSize, 
                                           args.kernelStd, 
                                           args.sigma))
        
        x_new = x_new - (args.nSubsets * bp + args.betaRecon * sqsNlm) / (normImg + args.betaRecon * 8)
        
    z = x_new + args.nesterov * (x_new - x)
    x = np.copy(x_new)
    
    # get loss function
    fp = reconNet.cDDFanProjection3d(x, angles) / projectorNorm
    dataLoss = 0.5 * np.sum(weight * (fp - prj / projectorNorm)**2)
    
    # nlm loss
    nlm = HYPR_NLM.NLM(x, ref, args.searchSize, args.kernelSize, args.kernelStd, args.sigma)
    nlm2 = HYPR_NLM.NLM(x**2, ref, args.searchSize, args.kernelSize, args.kernelStd, args.sigma)
    nlmLoss = np.sum(x**2 - 2 * x * nlm + nlm2)
    
    return x, z, dataLoss, nlmLoss


# In[15]:


header = ['lossData', 'lossNlm', 'rmseRoi']
vals = []

x_nesterov = np.copy(x)

# iteration
for iIter in range(args.nIters):
    if args.fixedGuide:
        guide = x0
    else:
        guide = x
    
    # SQS
    x, x_nesterov, dataLoss, nlmLoss =     SQSOneStep(reconNet, x, x_nesterov, guide, prjs, weights, normImg, projectorNorm, args, showPlots)
    
    rmse_roi = np.sqrt(np.mean((x - refs)[0, 128:-128, 128:-128, 0]**2))
    
    vals.append([dataLoss, nlmLoss, rmse_roi])
        
    if (iIter+1) % args.outputInterval == 0:
        if showPlots:
            display.clear_output()
            plt.figure(figsize=[18,6])
            plt.subplot(131); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(132); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(133); plt.imshow((x-refs)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.1, vmax=0.1)
            plt.show()
        
        print('%d: dataLoss = %g, nlmLoss = %g, rmse_roi = %g'%(iIter, dataLoss, nlmLoss, rmse_roi))


# In[20]:


if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
np.save(os.path.join(args.outDir, 'recon'), x)
np.savez(os.path.join(args.outDir, 'loss'), header = header, val = vals)
with open(os.path.join(args.outDir, 'args'), 'w') as f:
    for k in args.__dict__:
        f.write('%s = %s\n'%(k, str(args.__dict__[k])))


# In[21]:


if showPlots:
    plt.figure(figsize=[18,12])
    plt.subplot(231); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(232); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(233); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(234); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(235); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(236); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)


# In[ ]:




