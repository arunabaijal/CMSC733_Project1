"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    net = Img

    net = tf.layers.conv2d(net,64,kernel_size = 3,activation = tf.nn.relu)
    net = tf.layers.conv2d(net,64,kernel_size = 3,activation = tf.nn.relu)
    net = tf.nn.max_pool(net,ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1],padding = 'SAME')

    net = tf.layers.conv2d(net,64,kernel_size = 3,activation = tf.nn.relu)
    net = tf.layers.conv2d(net,64,kernel_size = 3,activation = tf.nn.relu)
    net = tf.nn.max_pool(net,ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1],padding = 'SAME')

    net = tf.layers.conv2d(net,128,kernel_size = 3,activation = tf.nn.relu)
    net = tf.layers.conv2d(net,128,kernel_size = 3,activation = tf.nn.relu)
    net = tf.nn.max_pool(net,ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1],padding = 'SAME')

    net = tf.layers.conv2d(net,128,kernel_size = 3,activation = tf.nn.relu)
    net = tf.layers.conv2d(net,128,kernel_size = 3,activation = tf.nn.relu)
    # maxpool here?

    x = tf.contrib.layers.flatten(net)

    x = tf.layers.dense(inputs=x, name='fc_1',units=1024, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc_2',units=8, activation=None)

    prLogits = x
    # prSoftMax = tf.nn.softmax(x)
    return prLogits

    # input_shape=(128, 128, 2)
    # input_img = Input(shape=input_shape)
     
    # x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv1', activation='relu')(input_img)
    # x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv2', activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    # x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv3', activation='relu')(x)
    # x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv4', activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
   
    # x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv5', activation='relu')(x)
    # x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv6', activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)
    
    # x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv7', activation='relu')(x)
    # x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv8', activation='relu')(x)
    
    # x = Flatten()(x)
    # x = Dense(1024, name='FC1')(x)
    # out = Dense(8, name='loss')(x)
    
    # model = Model(input=input_img, output=[out])
    # plot(model, to_file='HomegraphyNet_Regression.png', show_shapes=True)
    
    
    # model.compile(optimizer=Adam(lr=1e-3), loss=euclidean_distance)
    # return model

   

    return H4Pt

