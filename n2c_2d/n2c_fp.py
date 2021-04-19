#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import sys
import scipy
import glob


# In[2]:


sys.path.append('..')
import UNet
sys.path.append('/home/dwu/DeepRecon/ReconNet/python/')
import ReconNet
sys.path.append('/home/dwu/DeepRecon/')
import helper


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'noise2clean training fp')

# paths
parser.add_argument('--outDir', dest='outDir', type=str, default=None)
parser.add_argument('--name', dest='name', type=str, nargs='+', 
                    default=['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506'])
parser.add_argument('--testingName', dest='testingName', type=str, nargs='+', default=['L067', 'L096'])

parser.add_argument('--refDir', dest='refDir', type=str, 
                    default='/home/dwu/DeepRecon/train/data/real_2D_3_layer_mean_resolution_0.8/')
parser.add_argument('--testingPhantom', dest='testingPhantom', type=str, 
                    default='../../train/recon/data/fp/full_hann.npy')
parser.add_argument('--resolution', dest='resolution', type=float, default=0.8)
parser.add_argument('--paramFile', dest='paramFile', type=str, 
                    default='../../train/recon/data/fp/param.txt')
parser.add_argument('--nTestingSlicesPerPatient', dest='nTestingSlicesPerPatient', type=int, default=10)

# simulation
parser.add_argument('--N0', dest='N0', type=float, default=2e5)
parser.add_argument('--doseRate', dest='doseRate', type=float, default=0.25)
parser.add_argument('--filter', dest='filter', type=int, default=2, 
                    help='filter for fbp: 0-RL, 2-Hann')

# general network training
parser.add_argument('--device', dest='device', type=int, default=0)
parser.add_argument('--nEpochs', dest='nEpochs', type=int, default=100)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--nPatchesPerImg', dest='nPatchesPerImg', type=int, default=40)
parser.add_argument('--batchSize', dest='batchSize', type=int, default=40)
parser.add_argument('--outputInterval', dest='outputInterval', type=int, default=50)
parser.add_argument('--testInterval', dest='testInterval', type=int, default=25)

# data augmentation
parser.add_argument('--aug', dest='aug', type=int, default=1)
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


if sys.argv[0] != 'n2c_fp.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    showPlots = True
    args = parser.parse_args(['--device', '0',
                              '--dose', '0.125',
                              '--imgshape', '96', '96', '1',
                              '--name', 'L067', 'L096', 'L291',
                              '--testingName', 'L291', 'L096',
                              '--testInterval', '1',
                              '--outputInterval', '10',
                              '--lr', '1e-3',
#                               '--vmin', '-0.15',
#                               '--vmax', '1.15',
                              '--outDir', '../../train/recon/n2c_fp/test_train'
                              ])
else:
    showPlots = False
    args = parser.parse_args(sys.argv[1:])

    
for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[6]:


tf.reset_default_graph()
net = UNet.UNet()
net.FromParser(args)
net.BuildModel()

reconNet = ReconNet.ReconNet()
reconNet.FromFile(args.paramFile)
reconNet.dx = args.resolution
reconNet.dy = args.resolution

testingNet = UNet.UNet()
testingNet.FromParser(args)
testingNet.scope = net.scope + 'Test'
testingNet.imgshape = [reconNet.nx, reconNet.ny, 1]
testingNet.BuildModel()

weightCopiers = [tf.assign(r, v) for r,v in zip(tf.trainable_variables(testingNet.scope), 
                                                tf.trainable_variables(net.scope+'/'))]
learningRate = tf.placeholder(tf.float32, None, 'lr')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learningRate).minimize(net.loss)

saver = tf.train.Saver(max_to_keep=1)

if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)


# In[7]:


# read all the training slices
refs = []
resolutions = []
for name in args.name:
    if name not in args.testingName:
        img = np.load(os.path.join(args.refDir, '%s_full_hann.npy'%name))
        refs.append(img)

# read all the testing slices
refNames = [os.path.basename(s).split('_')[0] for s in glob.glob(os.path.join(args.refDir, '*_full_hann.npy'))]
testingRef = np.load(args.testingPhantom)
for name in args.testingName:
    ind = refNames.index(name)
    refs.append(testingRef[ind*args.nTestingSlicesPerPatient:(ind+1)*args.nTestingSlicesPerPatient, ...])

refs = np.concatenate(refs, 0) / args.imgNorm
nTestingSlices = args.nTestingSlicesPerPatient * len(args.testingName)


# In[8]:


# forward projection
reconNet.cSetDevice(args.device)
prjs = []

for iSlice in range(refs.shape[0]):
    prjs.append(reconNet.cDDFanProjection3d(np.copy(refs[[iSlice], ...], 'C')))
prjs = np.concatenate(prjs, 0)

# add noise to projections
np.random.seed(0)
if args.N0 > 0:
    prjs = prjs + np.sqrt((1 - args.doseRate) / args.doseRate * np.exp(prjs * args.imgNorm) / args.N0)     * np.random.normal(size = prjs.shape) / args.imgNorm


# In[9]:


# mask for network training
masks = helper.GetMasks2D(reconNet, [args.resolution] * refs.shape[0])


# In[10]:


# FBP
np.random.seed(0)
imgs = []

for i in range(prjs.shape[0]):
    if i % 10 == 0:
        print (i, end=', ')
    
    fp = prjs[[i],...]
    
    fsino = reconNet.cFilter3d(np.copy(fp[[0], ...], 'C'), args.filter)
    imgs.append(reconNet.cDDFanBackprojection3d(fsino, type_projector=1))
imgs = np.concatenate(imgs, 0)


# In[11]:


if showPlots:
    plt.figure(figsize=[12,6])
    plt.subplot(121); plt.imshow(refs[-nTestingSlices, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
    plt.subplot(122); plt.imshow(imgs[-nTestingSlices, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)


# In[12]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())


# In[13]:


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


# In[14]:


np.random.seed(0)
imgs *= masks

with open(os.path.join(args.outDir, 'args'), 'w') as f:
    for k in args.__dict__:
        f.write('%s = %s\n'%(k, str(args.__dict__[k])))

for epoch in range(args.nEpochs):
    # training code
    # extract patches for training images
    patchList = ExtractPatches([imgs[:-nTestingSlices, ...], 
                                refs[:-nTestingSlices, ...], 
                                masks[:-nTestingSlices, ...]], net.imgshape, args.nPatchesPerImg)
    inds = np.arange(patchList[0].shape[0])
    np.random.shuffle(inds)
    for i in range(len(patchList)):
        patchList[i] = patchList[i][inds]

    for i in range(0, patchList[0].shape[0], args.batchSize):
        inputImg = patchList[0][i:i+args.batchSize, ...]
        inputRef = patchList[1][i:i+args.batchSize, ...]
        inputMask = patchList[2][i:i+args.batchSize, ...]

        if args.aug:
            argOption = np.random.randint(4)
            inputImg = helper.Augmentation(inputImg, argOption)
            inputRef = helper.Augmentation(inputRef, argOption)
            inputMask = helper.Augmentation(inputMask, argOption)

        _, loss, recon = sess.run([train_step, net.loss, net.recon], 
                                  {net.img: inputImg + args.imgOffset, 
                                   net.ref: inputRef + args.imgOffset,
                                   net.mask: inputMask,
                                   learningRate: args.lr,
                                   net.training: True})
        recon -= args.imgOffset

        k = int(i / args.batchSize)
        if (k+1) % args.outputInterval == 0:
            print ('(%d, %d): %g'%(epoch, k, loss), flush=True)

        if (k+1) % (args.outputInterval * 5) == 0 and showPlots:
            _ = sess.run(weightCopiers)
            ind = np.random.randint(-nTestingSlices, 0)
            inputImg = imgs[[ind], ...]
            inputRef = refs[[ind], ...]
            inputMask = masks[[ind], ...]
            recon = sess.run(testingNet.recon, {testingNet.img: inputImg + args.imgOffset, 
                                                testingNet.training: False})
            recon -= args.imgOffset
            
            display.clear_output()
            plt.figure(figsize=[18,6])
            plt.subplot(131); plt.imshow(inputRef[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(132); plt.imshow(recon[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.subplot(133); plt.imshow(inputImg[0, 128:-128, 128:-128, 0].T, 'gray', vmin=args.vmin, vmax=args.vmax)
            plt.show()

    saver.save(sess, os.path.join(args.outDir, '%d'%epoch))

    # save intermediate results
    if (epoch + 1) % args.testInterval == 0 or (epoch + 1) == args.nEpochs:
        print ('Generating intermediate results')
        _ = sess.run(weightCopiers)

        res = []
        rmse_rois = []
        for iSlice in range(imgs.shape[0] - nTestingSlices, imgs.shape[0]):
            inputImg = imgs[[iSlice], ...]
            inputRef = refs[[iSlice], ...]
            inputMask = masks[[iSlice], ...]
            recon = sess.run(testingNet.recon, {testingNet.img: inputImg + args.imgOffset, 
                                                testingNet.training: False})
            recon -= args.imgOffset

            res.append(recon)
            rmse_rois.append(np.sqrt(np.mean((recon - inputRef)[0, 128:-128, 128:-128, 0]**2)))

        # save images
        res = np.concatenate(res, 0)
        rmse_rois = np.array(rmse_rois)
        np.save(os.path.join(args.outDir, 'recon'), res)
        np.savez(os.path.join(args.outDir, 'loss'), header='rmseRoi', val=rmse_rois)


# In[ ]:




