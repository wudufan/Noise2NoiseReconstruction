#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import sys
import scipy
import json


# In[2]:


sys.path.append('..')
import UNet
sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'noise2noise unsupervised denoising')

# paths
parser.add_argument('--outDir', dest='outDir', type=str, default=None)
parser.add_argument('--ref', dest='ref', type=str, 
                    default='../../train/recon/data/sino/full_hann.npy')
parser.add_argument('--paramFile', dest='paramFile', type=str, 
                    default='../../train/recon/data/sino/param.txt')
parser.add_argument('--resolutionFile', dest='resolutionFile', type=str, 
                    default='../../train/recon/data/sino/resolutions.npy')

# simulation
parser.add_argument('--N0', dest='N0', type=float, default=2e5)
parser.add_argument('--doseRate', dest='doseRate', type=float, default=0.25)
parser.add_argument('--filter', dest='filter', type=int, default=2, 
                    help='filter for fbp: 0-RL, 2-Hann')

# general network training
parser.add_argument('--device', dest='device', type=int, default=0)
parser.add_argument('--slices', dest='slices', type=int, nargs=2, default=[68, 69])
parser.add_argument('--preLr', dest='preLr', type=float, default=1e-3)

# general iteration
parser.add_argument('--checkPoint', dest='checkPoint', type=str, default=None)

# data augmentation
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)
parser.add_argument('--imgOffset', dest='imgOffset', type=float, default=-1)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[4]:


tf.reset_default_graph()
net = UNet.UNet()
parser = net.AddArgsToArgParser(parser)


# In[5]:


if sys.argv[0] != 'denoising_fp.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
#                               '--slices', '48', '49',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--dose', '0.125',
                              '--checkPoint', 
                              '../../train/recon/n2n_fp/pretrain/beta_2_filter_2_dose_0.125',
                              '--model', 'encoder_decoder',
                              '--outDir', '../../train/recon/n2n_fp/denoising/test'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[6]:


tf.reset_default_graph()

reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)

testingNet = UNet.UNet()
testingNet.FromParser(args)
testingNet.imgshape = [reconNet.nx, reconNet.ny, 1]
testingNet.BuildN2NModel()

saver = tf.train.Saver(var_list = tf.trainable_variables(testingNet.scope+'/'))

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

refs = x / args.imgNorm

# support only one resolution
resolution = np.load(args.resolutionFile)
resolution = np.mean(resolution[args.slices[0]:args.slices[1]])
reconNet.dx = resolution
reconNet.dy = resolution

if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[64]:


# forward projecting
reconNet.cSetDevice(args.device)
prjs = reconNet.cDDFanProjection3d(refs)

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


# In[65]:


# mask for network training
masks = helper.GetMasks2D(reconNet, [resolution] * refs.shape[0])


# In[66]:


# projection split
np.random.seed(0)
imgs1 = []
imgs2 = []

for i in range(prjs.shape[0]):
# for i in range(10):
    if i % 10 == 0:
        print (i, end=', ')
    
    inds = np.arange(0, prjs.shape[2], 2).astype(int)
    shift = np.random.randint(0, 2, inds.shape)
    set1 = inds + shift
    set2 = inds + (1 - shift)
    
    fp = prjs[[i],...]    
    
    fsino = reconNet.cFilter3d(np.copy(fp[[0], ...], 'C'), args.filter)
    img1 = reconNet.cDDFanBackprojection3d(np.copy(fsino[:, :, set1, :], 'C'), np.array(reconNet.angles)[set1], 
                                           type_projector=1) * 2
    img2 = reconNet.cDDFanBackprojection3d(np.copy(fsino[:, :, set2, :], 'C'), np.array(reconNet.angles)[set2], 
                                           type_projector=1) * 2
    
    imgs1.append(img1)
    imgs2.append(img2)
    
imgs1 = np.concatenate(imgs1, 0)
imgs2 = np.concatenate(imgs2, 0)


# In[67]:


if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(((imgs1+imgs2)/2)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[68]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())


# In[69]:


def Prediction(sess, imgs1, imgs2, refs, net, args, masks = None):
    if masks is None:
        masks = np.ones_like(imgs1)
    
    losses = []
    lossesN2n = []
    recons = []
    for iSlice in range(imgs1.shape[0]):
        loss, lossN2n, recon =         sess.run([net.loss, net.lossN2n, net.recon], 
                 {net.x1: imgs1[[iSlice], ...] + args.imgOffset, 
                  net.x2: imgs2[[iSlice], ...] + args.imgOffset, 
                  net.ref: refs[[iSlice], ...] + args.imgOffset,
                  net.mask: masks[[iSlice], ...],
                  net.training: False})
    
        recon -= args.imgOffset
        
        losses.append(loss)
        lossesN2n.append(lossN2n)
        recons.append(recon)
    
    return np.mean(losses), np.mean(lossN2n), np.concatenate(recons)


# In[70]:


saver.restore(sess, args.checkPoint)
_, _, x = Prediction(sess, imgs1, imgs2, (imgs1 + imgs2) / 2, testingNet, args)

rmse_roi = np.sqrt(np.mean((x - refs)[0, 128:-128, 128:-128, 0]**2))
print (rmse_roi)

if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)    


# In[19]:


# record
if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
np.save(os.path.join(args.outDir, 'recon'), x)
np.savez(os.path.join(args.outDir, 'loss'), header = 'rmseRoi', val = rmse_roi)
with open(os.path.join(args.outDir, 'args'), 'w') as f:
    for k in args.__dict__:
        f.write('%s = %s\n'%(k, str(args.__dict__[k])))


# In[20]:


if showPlots:
    plt.figure(figsize=[18,12])
    plt.subplot(231); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(232); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(233); plt.imshow(((imgs1 + imgs2) / 2)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(234); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(235); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(236); plt.imshow(((imgs1 + imgs2) / 2)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)


# In[ ]:




