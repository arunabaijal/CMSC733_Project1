#!/usr/bin/env python2
"""
CMSC 733 Porject 1
Created on Mon Feb 24 23:00:00 2020
@author: Ashwin Varghese Kuruttukulam
		 Aruna Baijal
"""
import tensorflow as tf
import cv2
import sys
import os
import glob
import math
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import  HomographyModel
from Network.Network import  Unsupervised_HomographyModel
import numpy as np
import argparse
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from utils.DataGenSup import genSup
from utils.DataGenUnsup import genUnsup
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import pickle

# Don't generate pyc codes
sys.dont_write_bytecode = True


def BatchGenSup(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,TrainingSampleSize):
	BatchInput = []
	BatchLabel = []
	
	ImageNum = 0
	while ImageNum < MiniBatchSize:
		RandIdx = random.randint(0, TrainingSampleSize-1)
		
		RandImagename=BasePath+'/data/'+str(RandIdx)+'.npz'        
		ImageNum += 1
		npzfile=np.load(RandImagename)
		image=npzfile['arr_0']
		##########################################################
		# Add any standardization or data augmentation here!
		##########################################################

		I1 = np.float32(image)
		I1=(I1-np.mean(I1))/255
		Label = BasePath+'/labels/'+str(RandIdx)+'.npz' 
		npzfile=np.load(Label)
		labelRegress=npzfile['arr_0']
		labelRegress.resize((8,1))
		labelRegress=labelRegress[:,0]
		# Append All Images and Mask
		BatchInput.append(I1)
		BatchLabel.append(labelRegress)
		
	return BatchInput, BatchLabel

def BatchGenUnsup(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,TrainingSampleSize):
	
	#get the basepath to the folder data/ where both train images and labels are present
	
	
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
	BatchInput - Batch of images
	BatchLabel - Batch of one-hot encoded labels 
	"""
	stackedDataBatch=[]
	IABatch = []
	cornerBatch = []
	
	ImageNum = 0
	while ImageNum < MiniBatchSize:
		# Generate random image
		RandIdx = random.randint(0, TrainingSampleSize-1)
		
		RandImagename=BasePath+'/unsupervised/data/'+str(RandIdx)+'.npz'        
		ImageNum += 1
		npzfile=np.load(RandImagename)
		image=npzfile['arr_0']
		##########################################################
		# Add any standardization or data augmentation here!
		##########################################################
		I1 = np.float32(image)
		I1=(I1-np.mean(I1))/255
		#print(I1.shape)
		
		RandIAname=BasePath+'/unsupervised/Ia/'+str(RandIdx)+'.npz'        
		npzfile=np.load(RandIAname)
		image=npzfile['arr_0']
		Ia = np.float32(image)
		Ia=(Ia-np.mean(Ia))/255
			  
		
		Label = BasePath+'/unsupervised/cornerData/'+str(RandIdx)+'.npz' 
		npzfile=np.load(Label)
		labelRegress=npzfile['arr_0']
		# Append All Images and Mask
		labelRegress.resize((8,1))
		labelRegress=labelRegress[:,0]
		stackedDataBatch.append(I1)

		#print(Ia.shape)	
		Ia.resize((128,128,1))		
		IABatch.append(Ia)
		cornerBatch.append(labelRegress)

	return stackedDataBatch, IABatch , cornerBatch


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


def TrainOperationUnsupervised(ImgPH,CornerPH,I2PH,DirNamesTrain, TrainLabels,NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType,TrainingSampleSize):
	
	pred_I2,I2 = Unsupervised_HomographyModel(ImgPH, CornerPH, I2PH,None, ImageSize, MiniBatchSize)

	with tf.name_scope('Loss'):
		loss = tf.reduce_mean(tf.abs(pred_I2 - I2))
	with tf.name_scope('Adam'):
		Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	EpochLossPH = tf.placeholder(tf.float32, shape=None)
	loss_summary = tf.summary.scalar('LossEveryIter', loss)
	epoch_loss_summary = tf.summary.scalar('LossPerEpoch', EpochLossPH)
	MergedSummaryOP1 = tf.summary.merge([loss_summary])
	MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])

	# Setup Saver
	Saver = tf.train.Saver(max_to_keep=NumEpochs)
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

		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			epoch_loss=0
			BatchLosses=[]
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				patchBatch, IABatch,cornerBatch= BatchGenUnsup(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,TrainingSampleSize)
				FeedDict = {ImgPH: patchBatch, CornerPH: cornerBatch, I2PH: IABatch}
				_, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
				BatchLosses.append(LossThisBatch)
				epoch_loss = epoch_loss + LossThisBatch

				# Tensorboard
				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
			  

			LossList.append(sum(BatchLosses)/len(BatchLosses))
			with open('TrainingLossData.pkl', 'wb') as f:
				pickle.dump([LossList], f)

			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved...')
			Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
			Writer.add_summary(Summary_epoch,Epochs)
			Writer.flush()

def TrainOperationSupervised(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType,TrainingSampleSize):
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
	H4pt = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

	with tf.name_scope('Loss'):
		###############################################
		# Fill your loss function of choice here!
		###############################################
		#LabelPH=tf.reshape(LabelPH,[MiniBatchSize,LabelPH.shape[1:4].num_elements()])
		shapeH4pt=tf.shape(H4pt)
		shapeLabel=tf.shape(LabelPH)
		loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4pt,LabelPH))))
	with tf.name_scope('Adam'):
		###############################################
		# Fill your optimizer of choice here!
		###############################################
		Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	tf.summary.scalar('LossEveryIter', loss)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operation
	MergedSummaryOP = tf.summary.merge_all()

	# Setup Saver
	Saver = tf.train.Saver(max_to_keep=NumEpochs)
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
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			appendAcc=[]
			BatchLosses=[]
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				BatchInput, BatchLabel = BatchGenSup(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,TrainingSampleSize)
				FeedDict = {ImgPH: BatchInput, LabelPH: BatchLabel}
				_, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
				BatchLosses.append(LossThisBatch)

				# Tensorboard
				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()
			
			LossList.append(sum(BatchLosses)/len(BatchLosses))
			with open('TrainingLossData_m.pkl', 'wb') as f:
				pickle.dump([LossList], f)
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
	Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--TrainingSampleSize', type=int, default=127, help='Number of examples to train on from the train images set')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	TrainingSampleSize=Args.TrainingSampleSize
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType

	# Setup all needed parameters including file reading
	DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	if (ModelType=='Unsup'):
		# Define PlaceHolder variables for Input and Predicted output
		cropSize=128
		rho=16
		#resize shape
		resize=(320,240)
		numTrainData=128
		numImagesLimit=5000
		#file path
		filePath=BasePath+'/Train/'
		if not os.path.exists(BasePath+''):
			os.makedirs(BasePath+'/unsupervised/')
		saveDest=BasePath+'/unsupervised/'
		
		generateImagesUnsupervised(cropSize,rho,resize,numTrainData,numImagesLimit,filePath,saveDest)		
		
		ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
		CornerPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8))
		I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128,1))
		
		TrainOperationUnsupervised(ImgPH,CornerPH,I2PH,DirNamesTrain, TrainLabels,126, ImageSize,
					   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
					   DivTrain, LatestFile, BasePath, LogsPath, ModelType,TrainingSampleSize)
	else:
		cropSize=128
		#range for perturbing the corners to get the homographies [-rho,+rho]
		rho=16
		#resize shape
		resize=(320,240)
		print('running supervise')
		#number of train sets to be made
		numTrainData=128
		#limit of the number of the images in the train set
		numImagesLimit=5000
		#file path
		filePath=BasePath+'/Train/'
		if not os.path.exists(BasePath+'/supervised/'):
			os.makedirs(BasePath+'/supervised/')
		saveDest=BasePath+'/supervised/'
		
		genSup(cropSize,rho,resize,numTrainData,numImagesLimit,filePath,saveDest)
		
		ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
		LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8)) # OneHOT labels
		BasePath=saveDest
		TrainOperationSupervised(ImgPH, LabelPH, DirNamesTrain, TrainLabels,40000, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType,TrainingSampleSize=TrainingSampleSize)
		
	
if __name__ == '__main__':
	main()
 

