#!/usr/bin/env python2
"""
CMSC 733 Porject 1
Created on Mon Feb 24 23:00:00 2020
@author: Ashwin Varghese Kuruttukulam
		 Aruna Baijal
"""
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os 
import time 


def saveData(savePath,data,i):
	if not os.path.exists(savePath+'data'):
		os.makedirs(savePath+'data')
	if not os.path.exists(savePath+'labels'):
		os.makedirs(savePath+'labels')
	np.savez(savePath+'data/'+str(i)+'.npz',data[0])
	np.savez(savePath+'labels/'+str(i)+'.npz',data[1])


def ImagesGen(cropSize,rho,resize,dataList,numTrainData,saveDest):
	for i in range(numTrainData):
		image=cv2.imread(dataList[i],cv2.IMREAD_GRAYSCALE)
		image=cv2.resize(image,resize)
		getLocX=random.randint(105,160)
		getLocY=random.randint(105,225)
		patchA=image[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]

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
		warped=np.asarray(cv2.warpPerspective(image,hBA,resize)).astype(np.uint8)
		#get the last patchB at the same location but on the warped image.
		patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]
		#stack images on top of each other.
		stackedData=np.dstack((patchA,patchB))
		#homogrpahy check
		saveData(saveDest,[stackedData,H4pts],i)


def loadTrainList(filePath,numTrainData,numImagesLimit):
	fileList=os.listdir(filePath)
	dataList=[]
	for i in range(numTrainData):
		randInt=random.randint(1,numImagesLimit-1)
		dataList.append(filePath+fileList[randInt])
	return dataList

#get the files
def genSup(cropSize,rho,resize,numTrainData,numImagesLimit,filePath,saveDest):
	files=loadTrainList(filePath,numTrainData,numImagesLimit)
	ImagesGen(cropSize,rho,resize,files,numTrainData,saveDest)
