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

    conv1=tf.layers.conv2d(Img,filters=64,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm1 = tf.layers.batch_normalization(conv1)
    
    
    conv2=tf.layers.conv2d(norm1,
                           filters=64,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm2 = tf.layers.batch_normalization(conv2)
    
    pool2=tf.layers.max_pooling2d(inputs=norm2,
                                  pool_size=(2,2),
                                  strides=2)
    
    #drop1=tf.layers.dropout(pool2,rate=0.5)

    conv3=tf.layers.conv2d(pool2,
                           filters=64,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm3 = tf.layers.batch_normalization(conv3)
    
    conv4=tf.layers.conv2d(norm3,
                           filters=64,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    
    pool3=tf.layers.max_pooling2d(inputs=conv4,
                                  pool_size=(2,2),
                                  strides=2)
    
    #drop2=tf.layers.dropout(pool3,rate=0.5)
    
    conv5=tf.layers.conv2d(pool3,
                           filters=128,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm4 = tf.layers.batch_normalization(conv5)

#    #pool4=tf.layers.max_pooling2d(inputs=norm4,
#                                  pool_size=(2,2),
#                                  strides=2)
    
    conv6=tf.layers.conv2d(norm4,
                           filters=128,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm5 = tf.layers.batch_normalization(conv6)

    pool4=tf.layers.max_pooling2d(inputs=norm5,
                                  pool_size=(2,2),
                                  strides=2)
    
    conv7=tf.layers.conv2d(pool4,
                           filters=128,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    
    #drop3=tf.layers.dropout(pool4,rate=0.5)
    norm6 = tf.layers.batch_normalization(conv7)
    
    conv8=tf.layers.conv2d(norm6,
                           filters=128,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    #now flattening the layer to add a fully connected layer
    flatLayer1=tf.reshape(conv8,[-1,conv8.shape[1:4].num_elements()])
    
    #adding a dense layer
    dense1=tf.layers.dense(inputs=flatLayer1,units=1024,activation=tf.nn.relu)
    
    #add dropout if required!
    H4Pt=tf.layers.dense(dense1,units=8,activation=None)
    
    return H4Pt

