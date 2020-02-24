#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import pickle


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
	"""
	Inputs: 
	BasePath - Path to images
	Outputs:
	ImageSize - Size of the Image
	DataPath - Paths of all images where testing will be run on
	"""
	DIR = BasePath+'/Test_Gen'
	nImages =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])   
	NumTrainSamples = nImages/2
	temp_img = cv2.imread(BasePath+'/Test_Gen/1_raw_image.jpg')
	ImageSize = (temp_img.shape[0],temp_img.shape[1],2*temp_img.shape[2])
	DataPath = []

	for k in range(NumTrainSamples):
		DataPath.append('Test_Gen/'+str(k+1))

	return ImageSize, DataPath


def SetupInputs(BasePath):
	DIR = BasePath+'/Test_Gen'
	nImages =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])   
	NumTrainSamples = nImages/2

	# reading labels pickle file
	with open(BasePath+'/labels_homography_train', 'rb') as f:
		dict_labels = pickle.load(f)
	TrainLabels = []
	DirNamesTrain = []
	temp_img = cv2.imread(BasePath+'/Test_Gen/1_raw_image.jpg')
	ImageSize = (temp_img.shape[0],temp_img.shape[1],2*temp_img.shape[2])
	for k in range(NumTrainSamples):
		TrainLabels.append(dict_labels[str(k+1)])
		DirNamesTrain.append('Test_Gen/'+str(k+1))
	NumClasses = 8
	SaveCheckPoint = 100
	return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses



def ReadImages(ImageSize, BasePath, dataPath):
	"""
	Inputs: 
	ImageSize - Size of the Image
	DataPath - Paths of all images where testing will be run on
	Outputs:
	I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
	I1 - Original I1 image for visualization purposes only
	"""
	
	RawImageName = BasePath + os.sep + dataPath + '_raw_image.jpg'   
	WarpedImageName = BasePath + os.sep + dataPath + '_warpped_image.jpg'
	
	I1_1 = np.float32(cv2.imread(RawImageName))
	I1_2 = np.float32(cv2.imread(WarpedImageName))
	I1 = np.dstack((I1_1, I1_2))
	
	if(I1 is None):
		# OpenCV returns empty list if image is not read! 
		print('ERROR: Image I1 cannot be read')
		sys.exit()
		
	##########################################################################
	# Add any standardization or cropping/resizing if used in Training here!
	##########################################################################

	# I1S = iu.StandardizeInputs(np.float32(I1))

	# I1Combined = np.expand_dims(I1S, axis=0)

	return [I1]
				

def TestOperation(BasePath,ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	ImageSize is the size of the image
	ModelPath - Path to load trained model from
	DataPath - Paths of all images where testing will be run on
	LabelsPathPred - Path to save predictions
	Outputs:
	Predictions written to ./TxtFiles/PredOut.txt
	"""
	# Length = ImageSize[0]
	# Predict output with forward pass, MiniBatchSize for Test is 1
	prLogits = HomographyModel(ImgPH, ImageSize, 1)

	# Setup Saver
	Saver = tf.train.Saver()
	with open('../Data/labels_homography_test', 'rb') as f:
		testing_labels = pickle.load(f)

	
	with tf.Session() as sess:
		Saver.restore(sess, ModelPath)
		print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
		
		OutSaveT = open(LabelsPathPred, 'w')

		for count in tqdm(range(np.size(DataPath))):       
			Img = ReadImages(ImageSize, BasePath, DataPath[count])
			print(DataPath[count])
			FeedDict = {ImgPH: Img}
			PredT = sess.run(prLogits, feed_dict=FeedDict)

			OutSaveT.write(str(PredT)+'\n')
			
		OutSaveT.close()


def ReadLabels(LabelsPathTest, LabelsPathPred):
	if(not (os.path.isfile(LabelsPathTest))):
		print('ERROR: Test Labels do not exist in '+LabelsPathTest)
		sys.exit()
	else:
		LabelTest = open(LabelsPathTest, 'r')
		LabelTest = LabelTest.read()
		LabelTest = map(float, LabelTest.split())

	if(not (os.path.isfile(LabelsPathPred))):
		print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
		sys.exit()
	else:
		LabelPred = open(LabelsPathPred, 'r')
		LabelPred = LabelPred.read()
		LabelPred = map(float, LabelPred.split())
		
	return LabelTest, LabelPred

		
def main():
	"""
	Inputs: 
	None
	Outputs:
	Prints out the confusion matrix with accuracy
	"""

	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
	Parser.add_argument('--BasePath', dest='BasePath', default='../Data', help='Path to load images from, Default:BasePath')
	Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
	Args = Parser.parse_args()
	ModelPath = Args.ModelPath
	BasePath = Args.BasePath
	LabelsPath = Args.LabelsPath

	# Setup all needed parameters including file reading
	ImageSize, DataPath = SetupAll(BasePath)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
	LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

	TestOperation(BasePath,ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred)

	# Plot Confusion Matrix
	# LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
	# print(LabelsTrue[0])
	# print(LabelsPred[0])
	# ConfusionMatrix(LabelsTrue, LabelsPred)
	 
if __name__ == '__main__':
	main()
 
