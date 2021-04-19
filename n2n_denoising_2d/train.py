#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import sys
import scipy


# In[2]:


sys.path.append('..')
import UNet
sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[23]:


import argparse
parser = argparse.ArgumentParser(
    description = 'pre-training for noise2noise, this will significantly accelerate convergence')

# paths
parser.add_argument('--checkPoint', dest='checkPoint', type=str, default=None)
parser.add_argument('--src', dest='src', type=str, 
                    default='../../train/recon/n2n_denoising_2d/gaussian_two_sets/recon')
parser.add_argument('--ref', dest='ref', type=str, default='../../train/recon/data/sino/full_gaussian.npy')
parser.add_argument('--resolution', dest='resolution', type=str, 
                    default='../../train/recon/data/sino/resolutions.npy')
parser.add_argument('--paramFile', dest='paramFile', type=str, 
                    default='../../train/recon/data/sino/param.txt')


# general network training
parser.add_argument('--device', dest='device', type=int, default=0)
parser.add_argument('--nSlices', dest='nSlices', type=int, default=10)
parser.add_argument('--preLr', dest='preLr', type=float, default=1e-3)
parser.add_argument('--nPatchesPerImg', dest='nPatchesPerImg', type=int, default=4)
parser.add_argument('--outputInterval', dest='outputInterval', type=int, default=50)

# general iteration
parser.add_argument('--nPreTrainingIters', dest='nPreTrainingIters', type=int, default=1000)

# data augmentation
parser.add_argument('--aug', dest='aug', type=int, default=1)
parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.019)
parser.add_argument('--imgOffset', dest='imgOffset', type=float, default=-1)

# window
parser.add_argument('--vmin', dest='vmin', type=float, default=0.84)
parser.add_argument('--vmax', dest='vmax', type=float, default=1.24)


# In[24]:


tf.reset_default_graph()
net = UNet.UNet()
parser = net.AddArgsToArgParser(parser)


# In[35]:


if sys.argv[0] != 'train.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
                              '--imgshape', '96', '96', '1',
                              '--beta', '0',
#                               '--src', '../../train/recon/n2n_denoising_fp/gaussian_two_sets/0.25/recon',
#                               '--ref', '../../train/recon/data/fp/full_hann_10.npy',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--nPreTrainingIters', '1000',
                              '--checkPoint', 
                              '../../train/recon/n2n_denoising_fp/pretrain/beta_0_dose_0.25',
                              '--model', 'encoder_decoder'])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[36]:


tf.reset_default_graph()
net = UNet.UNet()
net.FromParser(args)
net.BuildN2NModel()

learningRate = tf.placeholder(tf.float32, None, 'lr')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learningRate).minimize(net.loss)

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


saver = tf.train.Saver(var_list = tf.trainable_variables(net.scope+'/'))

if not os.path.exists(os.path.dirname(args.checkPoint)):
    os.makedirs(os.path.dirname(args.checkPoint))


# In[37]:


# read phantom
np.random.seed(0)
x = np.load(args.ref)
resolutions = np.load(args.resolution)

inds = np.arange(x.shape[0])
np.random.shuffle(inds)
inds = inds[:args.nSlices]
x = x[inds, ...]
resolutions = resolutions[inds[:args.nSlices]]

refs = x / args.imgNorm

imgs1 = np.load(args.src+'_0.npy')[inds,...] / args.imgNorm
imgs2 = np.load(args.src+'_1.npy')[inds,...] / args.imgNorm

if showPlots:
    plt.figure(figsize=[18,6])
    plt.subplot(131); plt.imshow(refs[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(132); plt.imshow(imgs1[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(133); plt.imshow(imgs2[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[38]:


# mask for network training
masks = helper.GetMasks2D(reconNet, resolutions)


# In[39]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())


# In[40]:


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
                  net.training: True})
    
        recon -= args.imgOffset
        
        losses.append(loss)
        lossesN2n.append(lossN2n)
        recons.append(recon)
    
    return np.mean(losses), np.mean(lossN2n), np.concatenate(recons)


# In[41]:


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


# In[42]:


# pre-training
np.random.seed(0)

for iIter in range(args.nPreTrainingIters):
    patches = ExtractPatches([imgs1, imgs2, masks], net.imgshape, args.nPatchesPerImg)

    if args.aug:
        aug = np.random.randint(4)
        for i in range(len(patches)):
            patches[i] = helper.Augmentation(patches[i], aug)

    _, loss = sess.run([train_step, net.loss], 
                       {net.x1: patches[0] + args.imgOffset, 
                        net.x2: patches[1] + args.imgOffset, 
                        net.ref: (patches[0] + patches[1]) / 2 + args.imgOffset, 
                        net.mask: patches[2], 
                        net.betaInput: args.beta,
                        net.training: True, 
                        learningRate: args.preLr})

    if (iIter + 1) % args.outputInterval == 0:
        print (iIter, loss)

    if (iIter + 1) % (args.outputInterval * 5) == 0 and showPlots:
        _ = sess.run(weightCopiers)
        _, _, recon = Prediction(sess, imgs1, imgs2, refs, testingNet, args, masks)

        rmse = np.sqrt(np.sum((recon - refs)**2 * masks) / np.sum(masks))
        iSlice = np.random.randint(imgs1.shape[0])

        display.clear_output()
        print ('rmse = %g'%rmse)
        plt.figure(figsize=[18,6])
        plt.subplot(131); plt.imshow(refs[iSlice, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
        plt.subplot(132); plt.imshow(recon[iSlice, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
        plt.subplot(133); plt.imshow(((imgs1+imgs2)/2)[iSlice, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
        plt.show()

saver.save(sess, args.checkPoint)


# In[ ]:




