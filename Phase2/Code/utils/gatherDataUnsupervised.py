#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:33:42 2019

@author: kartikmadhira
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os 
import time 

def loadTrainList(filePath,numTrainData,numImagesLimit):
    fileList=os.listdir(filePath)
    dataList=[]
    #take any random number and take 5000 of these images
    for i in range(numTrainData):
        randInt=random.randint(1,numImagesLimit-1)
        dataList.append(filePath+fileList[randInt])
    return dataList

def saveData(savePath,data,i):
    if not os.path.exists(savePath+'Ia'):
        os.makedirs(savePath+'Ia')
    if not os.path.exists(savePath+'cornerData'):
        os.makedirs(savePath+'cornerData')
    if not os.path.exists(savePath+'data'):
        os.makedirs(savePath+'data')
	
    #print(savePath+'data/'+str(i))
    #np.savez(savePath+'labels/'+str(i)+'.npz',data[1])
    np.savez(savePath+'cornerData/'+str(i)+'.npz',data[1])
    np.savez(savePath+'Ia/'+str(i)+'.npz',data[2])
    np.savez(savePath+'data/'+str(i)+'.npz',data[0])
    
    
def getImages(cropSize,rho,resize,dataList,numTrainData,saveDest):
    #load image
    for i in range(numTrainData):
        
        Ia=cv2.imread(dataList[i])
        Ia=cv2.resize(Ia,resize)
        image=cv2.imread(dataList[i],cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,resize)
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
        warped=np.asarray(cv2.warpPerspective(image,hBA,resize)).astype(np.uint8)
        #get the last patchB at the same location but on the warped image.
        patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]
        #stack images on top of each other.
        stackedData=np.dstack((patchA,patchB))
        Ia=patchB
        if(i%3000==0):
            print('Saved '+str(i)+' images')
        saveData(saveDest,[stackedData,pts2,Ia],i)

#get the files

def generateImagesUnsupervised(cropSize,rho,resize,numTrainData,numImagesLimit,filePath,saveDest):
    files=loadTrainList(filePath,numTrainData,numImagesLimit)
    getImages(cropSize,rho,resize,files,numTrainData,saveDest)
