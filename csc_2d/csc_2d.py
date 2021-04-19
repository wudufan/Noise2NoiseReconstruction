#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import scipy
import cupy
import sporco
import sporco.cupy
import sporco.cuda
from sporco import util
from sporco.dictlrn import cbpdndl
from sporco.cupy import np2cp, cp2np
from sporco.cupy.dictlrn import onlinecdl

sys_pipes = sporco.util.notebook_system_output()


# In[2]:


sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
import PriorFunctionSolver
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'csc reconstruction fp')

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
parser.add_argument('--filter', dest='filter', type=int, default=2, 
                    help='filter for fbp: 0-RL, 2-Hann')
parser.add_argument('--doseRate', dest='doseRate', type=float, default=0.25)

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

# csc
parser.add_argument('--kernelSize', dest='kernelSize', type=int, nargs=2, default = [10,10])
parser.add_argument('--nChannels', dest='nChannels', type=int, default = 32)
parser.add_argument('--lmbda', dest='lmbda', type=float, default = 0.005)
parser.add_argument('--nPreIter', dest='nPreIter', type=int, default = -1)
parser.add_argument('--nInnerIter', dest='nInnerIter', type=int, default = 250)
parser.add_argument('--checkPoint', dest='checkPoint', type=str, default = None)
# prefilter for csc
parser.add_argument('--lmbdaPre', dest='lmbdaPre', type=float, default = 1)


# data augmentation
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[4]:


if sys.argv[0] != 'csc_2d.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
#                               '--slices', '48', '49',
                              '--betaRecon', '0.02',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--nPreIter', '-1',
                              '--checkPoint', '../../train/recon/csc_fp/pretrain/kernel_10_channel_32_lmbda_0.005.npy',
                              '--outDir', '../../train/recon/csc_2d/test'])
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


# In[18]:


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


# In[19]:


if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[20]:


# training
if args.nPreIter > 0:
    cupy.cuda.Device(args.device).use()
    
    np.random.seed(0)
    
    s = np.transpose(imgs[..., 0], (1, 2, 0))
    sl, sh = util.tikhonov_filter(s, args.lmbdaPre)
    

    D0 = np.random.normal(size = args.kernelSize + [args.nChannels]).astype(np.float32)
#     opt = onlinecdl.OnlineConvBPDNDictLearn.Options({
#         'Verbose': True, 'ZeroMean': False, 'eta_a': 10.0,
#         'eta_b': 20.0, 'DataType': np.float32,
#         'CBPDN': {'rho': 5.0, 'AutoRho': {'Enabled': True},
#                   'RelaxParam': 1.8, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
#                   'FastSolve': False, 'DataType': np.float32}})
#     learner = onlinecdl.OnlineConvBPDNDictLearn(np2cp(D0), args.lmbdaTrain, opt)

#     learner.display_start()
#     for i in range(args.nPreIter):
#         ind = np.random.randint(imgs.shape[0])
#         buff = np2cp(pad(imgs[ind,...,0], args.kernelSize))
#         learner.solve(buff)
#     learner.display_end()

#     D1 = cp2np(learner.getdict())

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
    
    if args.checkPoint is not None:
        if not os.path.exists(os.path.dirname(args.checkPoint)):
            os.makedirs(os.path.dirname(args.checkPoint))
        np.save(args.checkPoint, D1)
else:
    D1 = np.load(args.checkPoint)


# In[21]:


if showPlots:
    plt.imshow(sporco.util.tiledict(D1))


# In[22]:


def csc(imgs, D, args, lmbda=None, opt = None):
    if lmbda is None:
        lmbda = args.lmbda
    
    if opt is None:
        opt = sporco.admm.cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': args.nInnerIter,
                                                  'HighMemSolve': True, 'RelStopTol': 5e-3,
                                                  'AuxVarObj': False})
    
    s = np.transpose(imgs[...,0], (1,2,0))
    sl, sh = util.tikhonov_filter(s, args.lmbdaPre)
    
    ys = []
    coefs = []
    for i in range(sh.shape[-1]):
        coef = sporco.cuda.cbpdn(D, sh[...,i], lmbda, opt, dev = args.device)
        y = np.sum(cp2np(sporco.cupy.linalg.fftconv(np2cp(D), np2cp(coef))), -1) + sl[...,i]
        
        coefs.append(coef)
        ys.append(y[np.newaxis, ..., np.newaxis])
    
    return np.concatenate(ys, 0), np.array(coefs)


# In[23]:


# test dictionary denoising
dimg, coefs = csc(imgs, D1, args)
print (0.5 * np.sum((dimg - imgs)**2), np.sum(np.abs(coefs)))


# In[24]:


if showPlots:
    d = dimg - imgs
    plt.figure(figsize=[18,6])
    plt.subplot(131); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(132); plt.imshow(dimg[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(133); plt.imshow(d[0, 128:-128, 128:-128, 0].T, 'gray')


# In[25]:


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


# In[26]:


weights = np.sqrt(np.exp(-prjs * args.imgNorm))
projectorNorm = CalcProjectorNorm(reconNet, weights)
normImg = reconNet.cDDFanNormImg3d(prjs, weights) / projectorNorm / projectorNorm
# masks = np.ones_like(masks)


# In[27]:


x = imgs
x0 = np.copy(x)

if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)

    rmse0_roi = np.sqrt(np.mean((x0 - refs)[0, 128:-128, 128:-128, 0]**2))
    print (rmse0_roi)


# In[28]:


def SQSOneStep(reconNet, x, x_nestrov, z, prj, weight, normImg, projectorNorm, args, verbose = 0):
    # projection term
    if not reconNet.rotview % args.nSubsets == 0:
        raise ValueError('reconNet.rotview cannot be divided by args.nSubsets')
    
    inds = helper.OrderedSubsetsBitReverse(reconNet.rotview, args.nSubsets)
    angles = np.array([reconNet.angles[i] for i in inds], np.float32)
    prj = prj[:, :, inds, :]
    weight = weight[:, :, inds, :]
    
    nAnglesPerSubset = int(reconNet.rotview / args.nSubsets)
    
    x_new = np.copy(x_nestrov)
    
    for i in range(args.nSubsets):
        if verbose:
            print ('set%d'%i, end=',', flush=True)
        curAngles = angles[i*nAnglesPerSubset:(i+1)*nAnglesPerSubset]
        curWeight = weight[:, :, i*nAnglesPerSubset:(i+1)*nAnglesPerSubset, :]
        fp = reconNet.cDDFanProjection3d(x_new, curAngles) / projectorNorm
        dprj = fp - prj[:, :, i*nAnglesPerSubset:(i+1)*nAnglesPerSubset, :] / projectorNorm
        bp = reconNet.cDDFanBackprojection3d(curWeight * dprj, curAngles) / projectorNorm
        
        x_new = x_new - (args.nSubsets * bp + args.betaRecon * (x_new - z)) / (normImg + args.betaRecon)
        
    x_nestrov = x_new + args.nesterov * (x_new - x)
    x = np.copy(x_new)
    
    # get loss function
    fp = reconNet.cDDFanProjection3d(x, angles) / projectorNorm
    dataLoss = 0.5 * np.sum(weight * (fp - prj / projectorNorm)**2)
    
    regLoss = 0.5 * np.sum((x - z)**2)
    
    return x, x_nestrov, dataLoss, regLoss  


# In[ ]:


header = ['lossData', 'lossReg', 'lossCoef', 'rmseRoi']
vals = []

x_nesterov = np.copy(x)
x_dict, _ = csc(x, D1, args)

# iteration
for iIter in range(args.nIters):
    # SQS
    x, x_nesterov, dataLoss, regLoss =     SQSOneStep(reconNet, x, x_nesterov, x_dict, prjs, weights, normImg, projectorNorm, args, showPlots)

    # dictionary
    x_dict, coefs = csc(x, D1, args)
    coefLoss = np.sum(np.abs(coefs))
    
    rmse_roi = np.sqrt(np.mean((x - refs)[0, 128:-128, 128:-128, 0]**2))
    
    vals.append([dataLoss, regLoss, coefLoss, rmse_roi])
    
    if (iIter+1) % args.outputInterval == 0:
        if showPlots:
            display.clear_output()
            plt.figure(figsize=[18,6])
            plt.subplot(131); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(132); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(133); plt.imshow(x0[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.show()
        
        print('%d: dataLoss = %g, regLoss = %g, coefLoss = %g, rmse_roi = %g'              %(iIter, dataLoss, regLoss, coefLoss, rmse_roi), flush=True)


# In[24]:


if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
np.save(os.path.join(args.outDir, 'recon'), x)
np.savez(os.path.join(args.outDir, 'loss'), header = header, val = vals)
with open(os.path.join(args.outDir, 'args'), 'w') as f:
    for k in args.__dict__:
        f.write('%s = %s\n'%(k, str(args.__dict__[k])))


# In[25]:


if showPlots:
    plt.figure(figsize=[18,12])
    plt.subplot(231); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(232); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(233); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(234); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(235); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(236); plt.imshow(imgs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)


# In[ ]:




