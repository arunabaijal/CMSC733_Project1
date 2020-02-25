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
import tensorflow as tf
from Network.Network import HomographyModel
import argparse 


def runTest(firstImage,ModelPath):
    #load image
    cropSize=128
    resize=(320,240)
    rho=16
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

    ImageSize = [128, 128, 2]
    ImgPH=tf.placeholder(tf.float32, shape=(1, 128, 128, 2))
    
    
    H4pt = HomographyModel(ImgPH, ImageSize, 1)
    Saver = tf.train.Saver()
  
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        Img=np.dstack((patchA,patchB))
        image=Img
        Img=np.array(Img).reshape(1,128,128,2)
        
        FeedDict = {ImgPH: Img}
        PredT = sess.run(H4pt,FeedDict)
    
    newPointsDiff=PredT.reshape(4,2)
    print(newPointsDiff)
    pts2=np.float32([[getLocY-cropSize/2,getLocX-cropSize/2],
          [getLocY+cropSize/2,getLocX-cropSize/2],
          [getLocY+cropSize/2,getLocX+cropSize/2],
          [getLocY-cropSize/2,getLocX+cropSize/2]])
    pts1=pts2+newPointsDiff
    H4pts=pts2-pts1
    hAB=cv2.getPerspectiveTransform(pts2,pts1)
    hBA=np.linalg.inv(hAB)


    cv2.line(firstI,   (pts1[0][0],pts1[0][1]),(pts1[1][0],pts1[1][1]), (0,0,255), 3)
    cv2.line(firstI,  (pts1[1][0],pts1[1][1]),(pts1[2][0],pts1[2][1]),(0,0,255), 3)
    cv2.line(firstI,  (pts1[2][0],pts1[2][1]),(pts1[3][0],pts1[3][1]),(0,0,255), 3)
    cv2.line(firstI,  (pts1[3][0],pts1[3][1]),(pts1[0][0],pts1[0][1]), (0,0,255), 3)
    
    cv2.imwrite('result'+'.png',firstI)


def main():
    Parser = argparse.ArgumentParser()
    
    Parser.add_argument('--Image', default='../Data/Train/141.jpg', help='Images')
    Parser.add_argument('--ModelPath', default='../Checkpoints/49model.ckpt', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    
    
    Args = Parser.parse_args()
    Image = Args.Image
    ModelPath = Args.ModelPath
    ModelType = Args.ModelType
    runTest(Image,ModelPath)

     
if __name__ == '__main__':
    main()

    
