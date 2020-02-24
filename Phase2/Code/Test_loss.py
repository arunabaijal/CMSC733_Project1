#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:05:35 2019

@author: kartikmadhira
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os 
import time 
import tensorflow as tf
from Network.Network import HomographyModel
import argparse 


def loadTrainList(filePath,numTrainData,numImagesLimit):
	fileList=os.listdir(filePath)
	dataList=[]
	#take any random number and take 5000 of these images
	for i in range(numTrainData):
		randInt=random.randint(1,numImagesLimit-1)
		dataList.append(filePath+fileList[randInt])
	return dataList


def saveData(savePath,data,i):
	if not os.path.exists(savePath+'data'):
		os.makedirs(savePath+'data')
	if not os.path.exists(savePath+'labels'):
		os.makedirs(savePath+'labels')
	np.savez(savePath+'data/'+str(i)+'.npz',data[0])
	np.savez(savePath+'labels/'+str(i)+'.npz',data[1])


def getImages(saveDest):
	#load image
	cropSize=128
	resize=(320,240)
	rho=16

	ImageSize = [128, 128, 2]
	ImgPH=tf.placeholder(tf.float32, shape=(1, 128, 128, 2))
	
	
	H4pt = HomographyModel(ImgPH, ImageSize, 1)
	Saver = tf.train.Saver()

	test_model_loss = []
  
	with tf.Session() as sess:
		for k in range(10):
			ModelPath = '../Checkpoints_sup_training/'+str(k)+'model.ckpt'
			Saver.restore(sess, ModelPath)
			print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
			images_save_dir = '../Data/Phase2_out/'+str(k)
			if not os.path.exists(images_save_dir):
	    		os.makedirs(images_save_dir)
			for i in range(20):
				firstImage = '../Data/Val/'+str(i+1)+'.jpg'
				firstI=cv2.imread(firstImage)
				
				image1=cv2.imread(firstImage,cv2.IMREAD_GRAYSCALE)
				image=cv2.resize(image1,resize)
				#get a random x and y location that does not have the borders
				#x is Y and y is X!
				getLocX=random.randint(105,160)
				getLocY=random.randint(105,225)
				#crop the image
				patchA=image[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]
				
				#perturb image randomly and apply homography
				pts1=np.float32([[getLocY-cropSize/2+random.randint(-rho,rho),getLocX-cropSize/2+random.randint(-rho,rho)],
				  [getLocY+cropSize/2+random.randint(-rho,rho),getLocX-cropSize/2+random.randint(-rho,rho)],
				  [getLocY+cropSize/2+random.randint(-rho,rho),getLocX+cropSize/2+random.randint(-rho,rho)],
				  [getLocY-cropSize/2+random.randint(-rho,rho),getLocX+cropSize/2+random.randint(-rho,rho)]])
				pts2=np.float32([[getLocY-cropSize/2,getLocX-cropSize/2],
				  [getLocY+cropSize/2,getLocX-cropSize/2],
				  [getLocY+cropSize/2,getLocX+cropSize/2],
				  [getLocY-cropSize/2,getLocX+cropSize/2]])

				H4pts=pts2-pts1
				
				#get the perspective transform
				hAB=cv2.getPerspectiveTransform(pts2,pts1)
				#get the inverse
				hBA=np.linalg.inv(hAB)
				#get the warped image from the inverse homography generated in the dataset
				warped=np.asarray(cv2.warpPerspective(image,hAB,resize)).astype(np.uint8)
				#get the last patchB at the same location but on the warped image.
				patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]
				
				
				cv2.line(firstI,   (pts1[0][0],pts1[0][1]),(pts1[1][0],pts1[1][1]), (255,0,0), 3)
				cv2.line(firstI,  (pts1[1][0],pts1[1][1]),(pts1[2][0],pts1[2][1]), (255,0,0), 3)
				cv2.line(firstI,  (pts1[2][0],pts1[2][1]),(pts1[3][0],pts1[3][1]), (255,0,0), 3)
				cv2.line(firstI,  (pts1[3][0],pts1[3][1]),(pts1[0][0],pts1[0][1]), (255,0,0), 3)
				
				Img=np.dstack((patchA,patchB))
				image=Img
				Img=np.array(Img).reshape(1,128,128,2)
				
				FeedDict = {ImgPH: Img}
				PredT = sess.run(H4pt,FeedDict)
				# print(PredT)
				# print(H4pts)
				#label=label.reshape(1,8)
				#print(PredT,label)
				#loss=np.sqrt(np.mean((PredT-label)**2))
				#print(loss)
				#plt.subplot(2,1,1)
				#plt.imshow(image[:,:,0])
				#plt.subplot(2,1,2)
				#plt.imshow(image[:,:,1])
		
				newPointsDiff=PredT.reshape(4,2)
				# print 
				test_model_loss.append(np.average(abs(newPointsDiff - H4pts)))
				# print('\n')
				# print(newPointsDiff)
				pts2=np.float32([[getLocY-cropSize/2,getLocX-cropSize/2],
					  [getLocY+cropSize/2,getLocX-cropSize/2],
					  [getLocY+cropSize/2,getLocX+cropSize/2],
					  [getLocY-cropSize/2,getLocX+cropSize/2]])
				pts1=pts2+newPointsDiff
				# H4pts=pts2-pts1
				#get the perspective transform
				hAB=cv2.getPerspectiveTransform(pts2,pts1)
				#get the inverse
				hBA=np.linalg.inv(hAB)
				#get the warped image from the inverse homography generated in the dataset
				#warped=np.asarray(cv2.warpPerspective(firstImageCol,hAB,)).astype(np.uint8)
				#get the last patchB at the same location but on the warped image.
				#patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]

				cv2.line(firstI,   (pts1[0][0],pts1[0][1]),(pts1[1][0],pts1[1][1]), (0,0,255), 3)
				cv2.line(firstI,  (pts1[1][0],pts1[1][1]),(pts1[2][0],pts1[2][1]),(0,0,255), 3)
				cv2.line(firstI,  (pts1[2][0],pts1[2][1]),(pts1[3][0],pts1[3][1]),(0,0,255), 3)
				cv2.line(firstI,  (pts1[3][0],pts1[3][1]),(pts1[0][0],pts1[0][1]), (0,0,255), 3)
				#plt.figure()
				#plt.imshow(warped)
				#plt.show()
				cv2.imwrite(images_save_dir+'/'+str(i)+'.png',firstI)
			epoch_loss =  sum(test_model_loss)/len(test_model_loss)
			print('epoch_loss',epoch_loss)
	#stack images on top of each other.
	#stackedData=np.dstack((patchA,patchB))
	# #homogrpahy check
	# orig=cv2.warpPerspective(patchB,hAB,(128,128))
	# plt.subplot(1,2,1)
	# plt.imshow(patchA)
	# plt.subplot(1,2,2)
	# plt.imshow(patchB)

def main():
	Parser = argparse.ArgumentParser()
	
	# Parser.add_argument('--ModelPath', default='../Checkpoints_sup_only_last/49model.ckpt', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	
	
	Args = Parser.parse_args()
	# ModelPath = Args.ModelPath
	ModelType = Args.ModelType
	#first='/home/kartikmadhira/CMSC733/YourDirectoryID_p1/Phase2/Data/Val/'+str(rand)+'.jpg'
	
	getImages('new')

	 
if __name__ == '__main__':
	main()

	
