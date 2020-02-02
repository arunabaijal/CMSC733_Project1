#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
# Add any python libraries here


'''
function to add standard deviation
'''
def normalize_features(featureArr):
	mean = np.mean(featureArr)
	sd = np.std(featureArr)
	featureArr = (featureArr-mean)/sd
	return featureArr

'''
feature vector generator
'''

def get_feature_vectors(img,corners):

	#parameters
	feature_crop_half_size = 20
	gaussian_blur_filter_size = 3
	resize_arr_size = 8

	#smoothing the image
	img = cv2.GaussianBlur(img,(gaussian_blur_filter_size,gaussian_blur_filter_size),cv2.BORDER_DEFAULT)

	# padding iamge by copying the border
	imgPadded = np.pad(img, (20,20), mode = 'edge')

	featureArrs = []
	for cornerInd in range(corners.shape[0]):
		pixelCoordinates = corners[cornerInd,:]

		# cropping image
		croppedImg = imgPadded[pixelCoordinates[0]:pixelCoordinates[0]+2*feature_crop_half_size,pixelCoordinates[1]:pixelCoordinates[1]+2*feature_crop_half_size]
		featureArr = cv2.resize(croppedImg, (resize_arr_size,resize_arr_size), interpolation = cv2.INTER_AREA)

		featureArr = featureArr.reshape((resize_arr_size*resize_arr_size,1))
		featureArr = normalize_features(featureArr)
		featureArrs.append(featureArr)
	return featureArrs

def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	corners = np.array([[20,20],[60,60],[150,150]])

	img1 = cv2.imread('../Data/Train/Set1/1.jpg',cv2.IMREAD_GRAYSCALE)
	featureVec = get_feature_vectors(img1)
	print(featureVec1)

	img2 = cv2.imread('../Data/Train/Set1/1.jpg',cv2.IMREAD_GRAYSCALE)
	featureVec2 = get_feature_vectors(img2)
	print(featureVec2)

	"""
	Feature Matching
	Save Feature Matching output as matching.png

	"""
	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    
if __name__ == '__main__':
    main()
 
