#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import tensorflow as tf
import numpy as np
import argparse


# In[4]:


class UNet:
    def __init__(self, imgshape = [640,640,1], scope='unet2d'):
        self.scope = scope
        self.imgshape = imgshape
        
        self.depth = 4
        self.nFilters = 32
        self.filterSz = [3,3,3]
        
        self.model = 'unet'
        
        self.bn = 0
        self.beta = 0
    
    def AddArgsToArgParser(self, parser):
        parser.add_argument('--scope', dest='scope', type=str, default='unet2d')
        parser.add_argument('--imgshape', dest='imgshape', type=int, nargs='+', default=[640,640,1])
        
        parser.add_argument('--nFilters', dest='nFilters', type=int, default=32)
        parser.add_argument('--filterSz', dest='filterSz', type = int, nargs='+', default = [3,3,3])
        parser.add_argument('--depth', dest='depth', type=int, default=4)
        
        parser.add_argument('--model', dest='model', type=str, default='unet')
        
        parser.add_argument('--bn', dest='bn', type=int, default=0)
        parser.add_argument('--beta', dest='beta', type=float, default=0)

        return parser
    
    def FromParser(self, args):
        for k in args.__dict__.keys():
            if k in self.__dict__.keys():
                setattr(self, k, args.__dict__[k])
    
    def Normalization(self, x, name = None):
        if self.bn:
            return tf.layers.batch_normalization(x, scale = False, training = self.training, name = name)
        else:
            return x
    
    def ModelResNet(self, x, reuse=False, scope=None):
        if scope is None:
            scope = self.scope
        
        with tf.variable_scope(scope, reuse = reuse):
            nFilters = self.nFilters
            x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='conv_pre')
            
            for i in range(self.depth):
                x1 = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='conv%d_0'%i)
                x1 = tf.nn.relu(self.Normalization(x1, 'bn%d_0'%i))
                x1 = tf.layers.conv2d(x1, nFilters, self.filterSz[:2], padding='same', name='conv%d_1'%i)
                x1 = self.Normalization(x1, 'bn%d_1'%i)
                x = tf.nn.relu(x + x1)
            
            return tf.layers.conv2d(x, self.imgshape[-1], self.filterSz[:2], padding='same', name='conv_final') 
    
    def ModelUNetDilated(self, x, reuse=False, scope=None):
        if scope is None:
            scope = self.scope
        
        with tf.variable_scope(scope, reuse = reuse):
            encodes = []
            for i in range(self.depth):
                if self.model == 'encoder_decoder':
                    dilation_rate = 1
                else:
                    dilation_rate = 2**i
                
                nFilters = self.nFilters
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', dilation_rate=dilation_rate,
                                     name='conv%d_down'%i)
                if i == 0:
                    x = tf.nn.relu(x)
                else:
                    x = tf.nn.relu(self.Normalization(x, 'bn%d_down'%i))
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', dilation_rate=dilation_rate, 
                                     name='conv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'bn%d_0'%i))
                encodes.append(x)
            
            for i in range(self.depth - 2, -1, -1):
                # typo: should not influence the results much, hold here to make the results replicable
#                 if self.model == 'encoder_decoder':
#                     dilation_rate = 1
#                 else:
                dilation_rate = 2**i
                
                nFilters = self.nFilters
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', dilation_rate=dilation_rate,
                                     name='tconv%d_up'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_up'%i))
                x = tf.concat((x, encodes[i]), -1)
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', dilation_rate=dilation_rate,
                                     name='tconv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_0'%i))
                
            return tf.layers.conv2d(x, self.imgshape[-1], self.filterSz[:2], padding='same', name='conv_final')
    
    def ModelUNet(self, x, reuse=False, scope=None):
        if scope is None:
            scope = self.scope
        
        with tf.variable_scope(scope, reuse = reuse):
            encodes = []
            for i in range(self.depth):
                nFilters = self.nFilters * (2**i)
                if i > 0:
                    x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], 2, padding='same', name='conv%d_down'%i)
                    x = tf.nn.relu(self.Normalization(x, 'bn%d_down'%i))
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='conv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'bn%d_0'%i))
                encodes.append(x)
            
            for i in range(self.depth - 2, -1, -1):
                nFilters = self.nFilters * (2**i)
                x = tf.layers.conv2d_transpose(x, nFilters, self.filterSz[:2], 2, padding='same', name='tconv%d_up'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_up'%i))
                x = tf.concat((x, encodes[i]), -1)
                x = tf.layers.conv2d_transpose(x, nFilters, self.filterSz[:2], padding='same', name='tconv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_0'%i))
                
            return tf.layers.conv2d(x, self.imgshape[-1], self.filterSz[:2], padding='same', name='conv_final')
    
    def ModelDIP(self, x, reuse=False, scope=None):
        if scope is None:
            scope = self.scope
        
        with tf.variable_scope(scope, reuse = reuse):
            encodes = []
            for i in range(self.depth):
                nFilters = self.nFilters * (2**i)
                if i > 0:
                    x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], 2, padding='same', name='conv%d_down'%i)
                    x = tf.nn.relu(self.Normalization(x, 'bn%d_down'%i))
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='conv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'bn%d_0'%i))
                encodes.append(x)
            
            for i in range(self.depth - 2, -1, -1):
                nFilters = self.nFilters * (2**i)
                x = tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2))
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='tconv%d_up'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_up'%i))
                x = x + encodes[i]
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='tconv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_0'%i))
                
            return tf.layers.conv2d(x, self.imgshape[-1], self.filterSz[:2], padding='same', name='conv_final')
    
    def BuildModel(self):
        with tf.variable_scope(self.scope):
            self.img = tf.placeholder(tf.float32, [None] + self.imgshape, 'img')
            self.ref = tf.placeholder(tf.float32, [None] + self.imgshape, 'ref')
            self.mask = tf.placeholder(tf.float32, [None] + self.imgshape, 'mask')
            self.training = tf.placeholder_with_default(False, None, 'training')
        
        if self.model.lower() == 'unet':
            model = self.ModelUNet
        elif self.model.lower() == 'unet_dilated' or self.model.lower() == 'encoder_decoder':
            model = self.ModelUNetDilated
        elif self.model.lower() == 'unet_dip':
            model = self.ModelDIP
        else:
            raise ValueError('model must one of "unet", "unet_dilated", "encode_decoder", or "unet_dip"')
        
        self.recon = model(self.img)
        self.loss = tf.sqrt(tf.reduce_sum((self.ref * self.mask - self.recon * self.mask)**2) / 
                            tf.reduce_sum(self.mask))
    
    # use external reference image for recon, e.g. SENSE reconstructed results
    def BuildN2NModel(self):
        with tf.variable_scope(self.scope):
            self.x1 = tf.placeholder(tf.float32, [None] + self.imgshape, 'x1')
            self.x2 = tf.placeholder(tf.float32, [None] + self.imgshape, 'x2')
            self.ref = tf.placeholder(tf.float32, [None] + self.imgshape, 'ref')
            self.mask = tf.placeholder(tf.float32, [None] + self.imgshape, 'mask')
            self.betaInput = tf.placeholder_with_default(tf.cast(self.beta, tf.float32), None, 'beta')
            self.training = tf.placeholder_with_default(False, None, 'training')
        
        if self.model.lower() == 'unet':
            model = self.ModelUNet
        elif self.model.lower() == 'unet_dilated' or self.model.lower() == 'encoder_decoder':
            model = self.ModelUNetDilated
        elif self.model.lower() == 'resnet':
            model = self.ModelResNet
        else:
            raise ValueError('model must one of "unet", "unet_dilated", "encoder_decoder" or "resnet"')
        
        self.y1 = model(self.x1)
        self.y2 = model(self.x2, reuse = True)
        
        self.recon = (self.y1 + self.y2) / 2
        
        self.lossN2n = (0.5 * tf.reduce_sum(self.mask * (self.y1 - self.x2)**2) + 
                        0.5 * tf.reduce_sum(self.mask * (self.y2 - self.x1)**2)) / tf.reduce_sum(self.mask)
        
        self.lossReg = tf.reduce_sum( self.mask * (self.recon - self.ref)**2) / tf.reduce_sum(self.mask)
        
        self.loss = self.lossN2n + self.lossReg * self.betaInput
        
    def BuildN2NModel2(self):
        with tf.variable_scope(self.scope):
            self.x1 = tf.placeholder(tf.float32, [None] + self.imgshape, 'x1')
            self.x2 = tf.placeholder(tf.float32, [None] + self.imgshape, 'x2')
            self.ref1 = tf.placeholder(tf.float32, [None] + self.imgshape, 'ref1')
            self.ref2 = tf.placeholder(tf.float32, [None] + self.imgshape, 'ref2')
            self.mask = tf.placeholder(tf.float32, [None] + self.imgshape, 'mask')
            self.training = tf.placeholder_with_default(False, None, 'training')
        
        if self.model.lower() == 'unet':
            model = self.ModelUNet
        elif self.model.lower() == 'unet_dilated':
            model = self.ModelUNetDilated
        else:
            raise ValueError('model must one of "unet", or "unet_dilated"')
        
        self.y1 = model(self.x1)
        self.y2 = model(self.x2, reuse = True)
        
        self.recon = (self.y1 + self.y2) / 2
        
        self.lossN2n = (0.5 * tf.reduce_sum(self.mask * (self.y1 - self.ref1)**2) + 
                        0.5 * tf.reduce_sum(self.mask * (self.y2 - self.ref2)**2)) / tf.reduce_sum(self.mask)
        
        self.loss = self.lossN2n


# In[5]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'UNet'])


# In[ ]:




