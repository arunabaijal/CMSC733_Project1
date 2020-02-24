#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
#from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import os, os.path
import pickle

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupInputs(BasePath):
	DIR = BasePath+'/Train_Gen'
	nImages =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])	
	NumTrainSamples = int(nImages/2)

	# reading labels pickle file
	with open(BasePath+'/labels_homography_train', 'rb') as f:
		dict_labels = pickle.load(f)
	TrainLabels = []
	DirNamesTrain = []
	temp_img = cv2.imread(BasePath+'/Train_Gen/1_raw_image.jpg')
	ImageSize = (temp_img.shape[0],temp_img.shape[1],2*temp_img.shape[2])
	for k in range(int(NumTrainSamples)):
		TrainLabels.append(dict_labels[str(k+1)])
		DirNamesTrain.append('Train_Gen/'+str(k+1))
	NumClasses = 8
	SaveCheckPoint = 100
	return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses


def euclidean_distance(y_true, y_pred):
	return (y_pred - y_true)
	
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
	"""
	Inputs: 
	BasePath - Path to COCO folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	ImageSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels 
	"""
	I1Batch = []
	LabelBatch = []
	
	ImageNum = 0
	while ImageNum < MiniBatchSize:
		# Generate random image
		RandIdx = random.randint(0, len(DirNamesTrain)-1)
		
		RandRawImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '_raw_image.jpg'   
		RandWarpedImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '_warpped_image.jpg'   
		ImageNum += 1
		
		##########################################################
		# Add any standardization or data augmentation here!
		##########################################################

		I1_1 = np.float32(cv2.imread(RandRawImageName))
		I1_2 = np.float32(cv2.imread(RandWarpedImageName))
		I1 = np.dstack((I1_1, I1_2))
		# Label = convertToOneHot(TrainLabels[RandIdx], 10)
		Label = TrainLabels[RandIdx]
		# print('len(TrainLabels)')
		# print(len(TrainLabels))
		# print('len(DirNamesTrain)')
		# print(len(DirNamesTrain))
		# # Append All Images and Mask
		# print(RandIdx)
		# print(I1)
		# print(Label)
		I1Batch.append(I1)
		LabelBatch.append(Label)
		
	return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)              

	
def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""      
	# Predict output with forward pass
	prLogits = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

	with tf.name_scope('Loss'):
		###############################################
		# Fill your loss function of choice here!
		###############################################
		# Change loss funciton to eucledian!!!
		# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = LabelPH, logits = prLogits)
		#loss = tf.nn.l2_loss(prLogits-LabelPH)
		loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(prLogits,LabelPH))))
		# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = LabelPH, logits = prLogits)
		# loss = tf.reduce_mean(cross_entropy)

	with tf.name_scope('Adam'):
		###############################################
		# Fill your optimizer of choice here!
		###############################################
		Optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(loss)

	# with tf.name_scope('Accuracy'):
	# 	prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
	# 	LabelDecoded = tf.argmax(LabelPH, axis=1)
	# 	Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
   

	# Tensorboard
	# Create a summary to monitor loss tensor
	tf.summary.scalar('LossEveryIter', loss)
	# tf.summary.scalar('Accuracy', Acc)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operationgoogel
	MergedSummaryOP = tf.summary.merge_all()

	# Setup Saver
	Saver = tf.train.Saver()
	
	with tf.Session() as sess:       
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
		
		LossList = [] 
		TrainingAccuracyList = [] 	
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			# BatchAccuracies = []
			BatchLosses = []
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
				# print('I1Batch')
				# print(I1Batch)
				# print('LabelBatch')
				# print(LabelBatch)
				FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
				# _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
				temp, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
				# batch_accuracy, temp, LossThisBatch, Summary = sess.run([Acc,Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
				# BatchAccuracies.append(float(batch_accuracy))
				BatchLosses.append(float(LossThisBatch))
				# Save checkpoint every some SaveCheckPoint's iterations
				# if PerEpochCounter % SaveCheckPoint == 0:
				# 	# Save the Model learnt in this epoch
				# 	SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
				# 	Saver.save(sess,  save_path=SaveName)
				# 	print('\n' + SaveName + ' Model Saved...')

				# Tensorboard
				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()

			LossList.append(sum(BatchLosses)/len(BatchLosses))
			# TrainingAccuracyList.append(sum(BatchAccuracies)/len(BatchAccuracies))
			
			with open('TrainingData.pkl', 'wb') as f:
				pickle.dump([LossList], f)

			#plt.plot(LossList)
			#plt.plot(TrainingAccuracyList)
			#plt.show()

			print('Epoch Loss = ',LossList[-1])
			# print('Epoch Accuracy = ',TrainingAccuracyList[-1])
			
			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved...')
			

def main():
	"""
	Inputs: 
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='../Data', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType

	# Setup all needed parameters including file reading
	# DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
	DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupInputs(BasePath)
	# SetupInputs(BasePath)


	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
	LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
	
	TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
		
	
if __name__ == '__main__':
	main()
 
