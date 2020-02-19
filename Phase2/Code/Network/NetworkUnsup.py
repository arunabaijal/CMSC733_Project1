#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:04:21 2019

@author: kartikmadhira
"""


import tensorflow as tf
import sys
import numpy as np
import tensorflow.contrib.slim as slim
from Misc.TFSpatialTransformer import transformer
from Misc.utils import *
# Don't generate pyc codes
sys.dont_write_bytecode = True


#class of the homography model that will return the model in the end

#inputs to the system will be: 
#1. Pa and Pb stacked depth wise
#2. Four corner points of the first image
#3. Original first image without taking a patch
#4. Mode of the network - Train or Test for dropout removal in test

class HomographyUnsupervised(object):        
    def __init__(self,stackedData,IA4points,Ia,mode,batchSize,patchSize,reuse_variables=None):
        self.stackedData=stackedData
        self.IA4points=IA4points
        self.Ia=Ia
        self.imgH=320
        self.imgW=240
        self.use_batch_norm=False
        self.mode=mode
        self.batchSize=batchSize
        self.I1=stackedData[:,:,:,0]
        self.I2=stackedData[:,:,:,1]
        self.is_training = reuse_variables
        self.reuse_variables = reuse_variables
        ##change the width and the height to self and remove from the homo parameters
        
        # Constants and variables used for spatial transformer
        M = np.array([[self.imgW/2.0, 0., self.imgW/2.0],
                  [0., self.imgH/2.0, self.imgH/2.0],
                  [0., 0., 1.]]).astype(np.float32)
        
        M_tensor  = tf.constant(M, tf.float32)
        self.M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [self.batchSize, 1,1])
        # Inverse of M
        M_inv = np.linalg.inv(M)
        M_tensor_inv = tf.constant(M_inv, tf.float32)
        self.M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [self.batchSize,1,1])
        y_t = tf.range(0, self.batchSize*self.imgH*self.imgW, self.imgW*self.imgH)
        z =  tf.tile(tf.expand_dims(y_t,[1]),[1,128*128])
        self.batch_indices_tensor = tf.reshape(z, [-1]) 
        #build the graph to output the H4pt points
        self.build_model()
        #self.solveDLT()
        self.transform()
        self.build_losses()
        
        
    def buildSupBlock(self):        
        with tf.variable_scope('conv_block1', reuse=self.reuse_variables): # H
            conv1 = self._conv_block(self.model_input, ([64, 64]), (3, 3), (1, 1),1)
            maxpool1 = self._maxpool2d(conv1, 2, 2) # H/2
        with tf.variable_scope('conv_block2', reuse=self.reuse_variables):
            conv2 = self._conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1),2)
            maxpool2 = self._maxpool2d(conv2, 2, 2) # H/4
        with tf.variable_scope('conv_block3', reuse=self.reuse_variables):
            conv3 = self._conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1),3)
            maxpool3 = self._maxpool2d(conv3, 2, 2) # H/8
        with tf.variable_scope('conv_block4', reuse=self.reuse_variables ):
            conv4 = self._conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1),4)
        # Dropout
        keep_prob = 0.5 if self.mode=='train' else 1.0
        dropout_conv4 = slim.dropout(conv4, keep_prob)
        # Flatten dropout_conv4
        out_conv_flat = slim.flatten(dropout_conv4)
        # Two fully-connected layers
        with tf.variable_scope('fc1'):
            fc1 = slim.fully_connected(out_conv_flat, 1024, scope='fc1')
            dropout_fc1 = slim.dropout(fc1, keep_prob)
        with tf.variable_scope('fc2'):
            fc2 = slim.fully_connected(dropout_fc1, 8, scope='fc2', activation_fn=None) #BATCH_SIZE x 8
        self.pred_h4p = fc2
        

    def solveDLT(self):
        #self.pts_1_tile = tf.reshape(self.IA4points, shape=(-1,8,1)) # BH_SIZE x 8 x 1
        batch_size = self.batchSize
        # Solve for H using DLT
        pred_h4p_tile = tf.expand_dims(self.pred_h4p, [2]) # BATCH_SIZE x 8 x 1
        # 4 points on the second image
        pred_pts_2_tile = tf.add(pred_h4p_tile, self.pts_1_til)
    
    
        # Auxiliary tensors used to create Ax = b equation
        M1 = tf.constant(Aux_M1, tf.float32)
        M1_tensor = tf.expand_dims(M1, [0])
        M1_tile = tf.tile(M1_tensor,[batch_size,1,1])
    
        M2 = tf.constant(Aux_M2, tf.float32)
        M2_tensor = tf.expand_dims(M2, [0])
        M2_tile = tf.tile(M2_tensor,[batch_size,1,1])
    
        M3 = tf.constant(Aux_M3, tf.float32)
        M3_tensor = tf.expand_dims(M3, [0])
        M3_tile = tf.tile(M3_tensor,[batch_size,1,1])
    
        M4 = tf.constant(Aux_M4, tf.float32)
        M4_tensor = tf.expand_dims(M4, [0])
        M4_tile = tf.tile(M4_tensor,[batch_size,1,1])
    
        M5 = tf.constant(Aux_M5, tf.float32)
        M5_tensor = tf.expand_dims(M5, [0])
        M5_tile = tf.tile(M5_tensor,[batch_size,1,1])
    
        M6 = tf.constant(Aux_M6, tf.float32)
        M6_tensor = tf.expand_dims(M6, [0])
        M6_tile = tf.tile(M6_tensor,[batch_size,1,1])
    
    
        M71 = tf.constant(Aux_M71, tf.float32)
        M71_tensor = tf.expand_dims(M71, [0])
        M71_tile = tf.tile(M71_tensor,[batch_size,1,1])
    
        M72 = tf.constant(Aux_M72, tf.float32)
        M72_tensor = tf.expand_dims(M72, [0])
        M72_tile = tf.tile(M72_tensor,[batch_size,1,1])
    
        M8 = tf.constant(Aux_M8, tf.float32)
        M8_tensor = tf.expand_dims(M8, [0])
        M8_tile = tf.tile(M8_tensor,[batch_size,1,1])
    
        Mb = tf.constant(Aux_Mb, tf.float32)
        Mb_tensor = tf.expand_dims(Mb, [0])
        Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])
    
        # Form the equations Ax = b to compute H
        # Form A matrix
        A1 = tf.matmul(M1_tile, self.pts_1_til) # Column 1
        A2 = tf.matmul(M2_tile, self.pts_1_til) # Column 2
        A3 = M3_tile                   # Column 3
        A4 = tf.matmul(M4_tile, self.pts_1_til) # Column 4
        A5 = tf.matmul(M5_tile, self.pts_1_til) # Column 5
        A6 = M6_tile                   # Column 6
        A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, self.pts_1_til)# Column 7
        A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, self.pts_1_til)# Column 8
    
        A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                       tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                       tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
             tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
        # Form b matrix
        b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    
        # Solve the Ax = b
        H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    
   
        # Add ones to the last cols to reconstruct H for computing reprojection error
        h_ones = tf.ones([batch_size, 1, 1])
        H_9el = tf.concat([H_8el,h_ones],1)
        H_flat = tf.reshape(H_9el, [-1,9])
        self.H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3
       
    
    def transform(self):
        # Transform H_mat since we scale image indices in transformer
        H_mat = tf.matmul(tf.matmul(self.M_tile_inv, self.H_mat), self.M_tile)
        # Transform image 1 (large image) to image 2
        out_size = (self.imgH, self.imgW)
        I12 = tf.slice(self.stackedData,[0,0,0,0],[self.batchSize,128,128,1])
        print(str(tf.shape(self.stackedData))+'================================')
        warped_images, _ = transformer(I12, H_mat, out_size)
        # TODO: warp image 2 to image 1
    
        warped_gray_images = tf.reduce_mean(warped_images, 3)
#        warped_images_flat = tf.reshape(warped_gray_images, [-1])
#        self.getPatches()
#        pixel_indices =  self.patch_indices_flat + self.batch_indices_tensor
#        pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)
        self.pred_I2 = tf.reshape(warped_gray_images, [self.batchSize, 128, 128, 1])
    
    
    def build_model(self):
    # Declare types of activation function, weight_initialization of conv layer. We can set for each conv by setting locally later
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training), \
            slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
                with tf.variable_scope('model', reuse=self.reuse_variables):
                    self.model_input = self.stackedData
                    self.pts_1_tile=self.IA4points
                    self.pts_1_til = tf.reshape(self.IA4points, shape=(-1,8,1)) # BH_SIZE x 8 x 1
                    self.buildSupBlock()
                    #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))

                    self.solveDLT()
       
    def build_losses(self):
        I2 = self.I2
        self.l1_loss = tf.reduce_mean(tf.abs(self.pred_I2 - I2))
    
    
    def _conv2d(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, scope=''):
        p = np.floor((kernel_size -1)/2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        out_conv =  slim.conv2d(inputs=p_x, num_outputs=num_out_layers, kernel_size=kernel_size, stride=stride, padding="VALID", activation_fn=activation_fn, scope=scope)
        if self.use_batch_norm:
            out_conv = slim.batch_norm(out_conv, self.is_training)
        return out_conv

    def _conv_block(self, x, num_out_layers, kernel_sizes, strides,scopeNum):
        conv1 = self._conv2d(x, num_out_layers[0], kernel_sizes[0], strides[0], scope='conv1'+str(scopeNum))
        conv2 = self._conv2d(conv1, num_out_layers[1], kernel_sizes[1], strides[1], scope='conv2'+str(scopeNum))
        return conv2

    def _maxpool2d(self, x, kernel_size, stride):
        p = np.floor((kernel_size -1)/2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size, stride=stride)
    

    def getPatches(self):
        x_t_flat, y_t_flat = self.get_mesh_grid_per_img(128, 128)
        x_start_tf = self.IA4points[:,0] # 1,
        y_start_tf = self.IA4points[:,1] # (1, )
        print(y_t_flat,y_start_tf)
        patch_indices_tf = (y_t_flat + y_start_tf)*self.imgW + (x_t_flat + x_start_tf)     
        patch_indices_tf = tf.cast(patch_indices_tf, tf.int32)
        self.patch_indices_flat = tf.reshape(patch_indices_tf, [-1])


        
    def get_mesh_grid_per_img(self,c_w, c_h, x_start=0, y_start=0):
      """Get 1D array of indices of pixels in the image of size c_h x c_w"""
      x_t = tf.matmul(tf.ones([c_h, 1]),
                tf.transpose(\
                    tf.expand_dims(\
                        tf.linspace(tf.cast(x_start,'float32'), tf.cast(x_start+c_w-1,'float32'), c_w), 1), [1,0]))
      y_t = tf.matmul(tf.expand_dims(\
                        tf.linspace(tf.cast(y_start,'float32'), tf.cast(y_start+c_h-1,'float32'), c_h), 1),
                      tf.ones([1, c_w]))
      x_t_flat = tf.reshape(x_t, [-1]) # 1 x D
      y_t_flat = tf.reshape(y_t, [-1])
      return  x_t_flat, y_t_flat
