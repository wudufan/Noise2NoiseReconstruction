#!/usr/bin/env python
# coding: utf-8

# In[21]:


import tensorflow as tf
import numpy as np
import os
import sys
import scipy
import json
import copy


# In[22]:


sys.path.append('..')
import UNet
sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[23]:


import argparse
parser = argparse.ArgumentParser(description = 'noise2noise unsupervised reconstruction forward projection')

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
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--preLr', dest='preLr', type=float, default=1e-3)
parser.add_argument('--batchSize', dest='batchSize', type=int, default=40)
parser.add_argument('--nPatchesPerImg', dest='nPatchesPerImg', type=int, default=40)
parser.add_argument('--outputInterval', dest='outputInterval', type=int, default=5)

# general iteration
parser.add_argument('--nIters', dest='nIters', type=int, default=100)
parser.add_argument('--nSubsets', dest='nSubsets', type=int, default=12)
parser.add_argument('--nesterov', dest='nesterov', type=float, default=0.5)
parser.add_argument('--nSubIters', dest='nSubIters', type=int, default=5)
parser.add_argument('--nPreTrainingIters', dest='nPreTrainingIters', type=int, default=-1)
parser.add_argument('--checkPoint', dest='checkPoint', type=str, default=None)
parser.add_argument('--gamma', dest='gamma', type=float, default = 7.5e-4)
parser.add_argument('--betaRecon', dest='betaRecon', type=float, default = 5)
# parser.add_argument('--betaGaussian', dest='betaGaussian', type=float, default = 2.5e-4)
# parser.add_argument('--nGaussianIters', dest='nGaussianIters', type=int, default = 100)

# data augmentation
parser.add_argument('--aug', dest='aug', type=int, default=0)
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)
parser.add_argument('--imgOffset', dest='imgOffset', type=float, default=-1)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[24]:


tf.reset_default_graph()
net = UNet.UNet()
parser = net.AddArgsToArgParser(parser)


# In[5]:


if sys.argv[0] != 'recon_alter_fp.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
#                               '--slices', '0', '1',
                              '--imgshape', '96', '96', '1',
                              '--gamma', '7.5e-4',
                              '--dose', '0.125',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--checkPoint', '../../train/recon/n2n_fp/pretrain/beta_0_filter_2_dose_0.125',
                              '--model', 'encoder_decoder', '--depth', '4',
                              '--nIters', '100',
                              '--outDir', '../../train/recon/n2n_fp/test'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[6]:


tf.reset_default_graph()
net = UNet.UNet()
net.FromParser(args)
net.BuildN2NModel()

reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)
reconNet.cSetDevice(args.device)

testingNet = UNet.UNet()
testingNet.FromParser(args)
testingNet.scope = net.scope + 'Test'
testingNet.imgshape = [reconNet.nx, reconNet.ny, 1]
testingNet.BuildN2NModel()

weightCopiers = [tf.assign(r, v) for r,v in zip(tf.trainable_variables(testingNet.scope), 
                                                tf.trainable_variables(net.scope+'/'))]
learningRate = tf.placeholder(tf.float32, None, 'lr')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learningRate).minimize(net.loss)

saver = tf.train.Saver(var_list = tf.trainable_variables(net.scope+'/'))

if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
if args.checkPoint is not None:
    if not os.path.exists(os.path.dirname(args.checkPoint)):
        os.makedirs(os.path.dirname(args.checkPoint))


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


# In[8]:


# forward projecting
reconNet.cSetDevice(args.device)
prjs = reconNet.cDDFanProjection3d(refs)

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


# In[9]:


# mask for network training
masks = helper.GetMasks2D(reconNet, [resolution] * refs.shape[0])


# In[10]:


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


# In[11]:


weights = np.sqrt(np.exp(-prjs * args.imgNorm))
projectorNorm = CalcProjectorNorm(reconNet, weights)
normImg = reconNet.cDDFanNormImg3d(prjs, weights) / projectorNorm / projectorNorm
# masks = np.ones_like(masks)


# In[12]:


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


# In[13]:


if showPlots:
    plt.figure(figsize=[12,6])
    plt.subplot(121); plt.imshow(((imgs1+imgs2)/2)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(122); plt.imshow(((imgs1 + imgs2)/2 - refs)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.1, vmax=0.1)


# In[14]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())


# In[15]:


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


# In[16]:


def ExtractPatches(imgList, patchSize, nPatchesPerImg):
    imgshape = imgList[0].shape
    
    ixs = np.random.randint(0, imgshape[1] - patchSize[0] + 1, imgshape[0] * nPatchesPerImg)
    iys = np.random.randint(0, imgshape[2] - patchSize[1] + 1, imgshape[0] * nPatchesPerImg)
    
    patchList = []
    for img in imgList:
        patches = np.zeros([img.shape[0] * nPatchesPerImg, patchSize[0], patchSize[1], 1], np.float32)
        for i, (ix, iy) in enumerate(zip(ixs, iys)):
            patches[i, ..., 0] = img[int(i / nPatchesPerImg), ix:ix+patchSize[0], iy:iy+patchSize[1], 0]
        patchList.append(patches)
    
    return patchList


# In[17]:


# pre-training
np.random.seed(0)
if args.nPreTrainingIters <= 0:
    saver.restore(sess, args.checkPoint)
else:
    for iIter in range(args.nPreTrainingIters):
        patches = ExtractPatches([imgs1, imgs2, masks], net.imgshape, args.nPatchesPerImg)
        
        _, loss = sess.run([train_step, net.loss], 
                           {net.x1: patches[0] + args.imgOffset, 
                            net.x2: patches[1] + args.imgOffset, 
                            net.ref: (patches[0] + patches[1]) / 2 + args.imgOffset, 
                            net.mask: patches[2], 
                            net.betaInput: args.beta,
                            net.training: True, 
                            learningRate: args.preLr})

        if (iIter + 1) % 5 == 0:
            print (iIter, loss)

        if (iIter + 1) % 25 == 0:
            _ = sess.run(weightCopiers)
            _, _, recon = Prediction(sess, imgs1, imgs2, refs, testingNet, args, masks)
            
            rmse = np.sqrt(np.sum((recon - refs)**2 * masks) / np.sum(masks))
            iSlice = np.random.randint(imgs1.shape[0])
            
            if showPlots:
                display.clear_output()
                print ('rmse = %g'%rmse)
                plt.figure(figsize=[18,6])
                plt.subplot(131); plt.imshow(refs[iSlice, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
                plt.subplot(132); plt.imshow(recon[iSlice, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
                plt.subplot(133); plt.imshow(((imgs1+imgs2)/2)[iSlice, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
                plt.show()
    
    if args.checkPoint is not None:
        saver.save(sess, args.checkPoint)


# In[18]:


_ = sess.run(weightCopiers)
_, _, z = Prediction(sess, imgs1, imgs2, (imgs1 + imgs2) / 2, testingNet, args)
x = np.copy(z)
# x = (imgs1 + imgs2) / 2
x0 = np.copy(x)

if showPlots:
    plt.figure(figsize=[6,6])
    plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    rmse0_roi = np.sqrt(np.mean((x0 - refs)[0, 128:-128, 128:-128, 0]**2))
    print (rmse0_roi)


# In[19]:


def SQSOneStep(reconNet, x, x_nesterov, z, masks, prj, weight, normImg, projectorNorm, args, verbose = 0):
    # projection term
    if not reconNet.rotview % args.nSubsets == 0:
        raise ValueError('reconNet.rotview cannot be divided by args.nSubsets')
        
    inds = helper.OrderedSubsetsBitReverse(reconNet.rotview, args.nSubsets)
    angles = np.array([reconNet.angles[i] for i in inds], np.float32)
    prj = prj[:,:, inds, :]
    weight = weight[:, :, inds, :]
    
    nAnglesPerSubset = int(reconNet.rotview / args.nSubsets)
    
    x_new = np.copy(x_nesterov)
    
    for i in range(args.nSubsets):
        if verbose:
            print ('set%d'%i, end=',', flush=True)
        curAngles = angles[i*nAnglesPerSubset:(i+1)*nAnglesPerSubset]
        curWeight = weight[:, :, i*nAnglesPerSubset:(i+1)*nAnglesPerSubset, :]
        fp = reconNet.cDDFanProjection3d(x_new, curAngles) / projectorNorm
        dprj = fp - prj[:, :, i*nAnglesPerSubset:(i+1)*nAnglesPerSubset, :] / projectorNorm
        bp = reconNet.cDDFanBackprojection3d(curWeight * dprj, curAngles) / projectorNorm
        
        x_new = x_new - (args.nSubsets * bp + args.gamma * args.betaRecon * (x_new - z) * masks) /         (normImg + args.gamma * args.betaRecon * masks)
        
    x_nesterov = x_new + args.nesterov * (x_new - x)
    x = np.copy(x_new)
    
    # get loss function
    fp = reconNet.cDDFanProjection3d(x, angles) / projectorNorm
    dataLoss = np.sum(weight * (fp - prj / projectorNorm)**2)
    
    return x, x_nesterov, dataLoss


# In[20]:


# cost function storage
header = ['lossData', 'lossReg', 'lossN2n', 'rmseRoi']
vals = []

x_nesterov = np.copy(x)

# iteration
for iIter in range(args.nIters):
    # SQS
    x, x_nesterov, lossData = SQSOneStep(reconNet, x, x_nesterov, z, masks, prjs, weights, normImg, projectorNorm, args, 
                                         verbose = showPlots)
    
    # network
    for iSubIter in range(args.nSubIters):
        patches = ExtractPatches([imgs1, imgs2, x, masks], net.imgshape, args.nPatchesPerImg)
        _ = sess.run(train_step, 
                             {net.x1: patches[0] + args.imgOffset, 
                              net.x2: patches[1] + args.imgOffset, 
                              net.ref: patches[2] + args.imgOffset, 
                              net.mask: patches[3], 
                              net.betaInput: args.betaRecon,
                              net.training: True, 
                              learningRate: args.lr})
    
    # update
    _ = sess.run(weightCopiers)
    loss, lossN2n, z = Prediction(sess, imgs1, imgs2, refs, testingNet, args, masks)
    
    lossReg = np.sum((x - z)**2 * masks)
    rmse_roi = np.sqrt(np.mean((x - refs)[0, 128:-128, 128:-128, 0]**2))
    
    vals.append([lossData, lossReg, lossN2n * imgs1.size, rmse_roi])
    
    if (iIter + 1) % args.outputInterval == 0:
        if showPlots:
            display.clear_output()
            plt.figure(figsize=[18,6])
            plt.subplot(131); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(132); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(133); plt.imshow((x-refs)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.1, vmax=0.1)
            plt.show()
        
        print('%d: lossData = %g, lossReg = %g, lossN2n = %g, rmse_roi = %g'                  %(iIter, lossData, lossReg, lossN2n * imgs1.size, rmse_roi), flush=True)


# In[ ]:


# record
if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)
np.save(os.path.join(args.outDir, 'recon'), x)
np.savez(os.path.join(args.outDir, 'loss'), header = header, val = vals)
with open(os.path.join(args.outDir, 'args'), 'w') as f:
    for k in args.__dict__:
        f.write('%s = %s\n'%(k, str(args.__dict__[k])))


# In[ ]:


if showPlots:
    plt.figure(figsize=[18,12])
    plt.subplot(231); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(232); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(233); plt.imshow(((imgs1 + imgs2) / 2)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(234); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(235); plt.imshow(x[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)
    plt.subplot(236); plt.imshow(((imgs1 + imgs2) / 2)[0, 128:-128, 128:-128, 0].T, 'gray', vmin=-0.15, vmax=1.15)


# In[ ]:




