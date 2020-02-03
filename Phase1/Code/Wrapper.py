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
	featureArr = (featureArr-mean)/(sd+10**-7)
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
		featureArr = imgPadded[pixelCoordinates[0]:pixelCoordinates[0]+2*feature_crop_half_size,pixelCoordinates[1]:pixelCoordinates[1]+2*feature_crop_half_size]
		featureArr = cv2.resize(featureArr, (resize_arr_size,resize_arr_size), interpolation = cv2.INTER_AREA)
		featureArr = featureArr.reshape((resize_arr_size*resize_arr_size,1))
		featureArr = normalize_features(featureArr)
		featureArrs.append(featureArr)

	return featureArrs

def find_second_smallest(arr):
	minVal = min(arr)[0]
	arr.remove(minVal)
	minVal = min(arr)[0]
	return minVal

'''
feature matching function
'''
def match_features(featureVec1,featureVec2):
	# print(featureVec1[0][0])
	# print(featureVec2[0][0])
	matches = []
	matching_ratio_threshold = 0.5
	for vec1_ind,vec1 in enumerate(featureVec1):
		sum_sq_dist = []
		for vec2 in featureVec2:
			# print(vec1)
			# print(vec2)
			print(sum((vec1-vec2)**2))
			# quit()
			sum_sq_dist.append(sum((vec1-vec2)**2))
		min_val = min(sum_sq_dist)[0]
		second_min_val = find_second_smallest(sum_sq_dist)
		print('matching ratio for corner ',str(vec1_ind),'is = ',(min_val/second_min_val))
		if((min_val/second_min_val)<matching_ratio_threshold):
			vec2_ind = sum_sq_dist.index(min_val)
			matches.append([vec1_ind,vec2_ind])
	return matches

# def print_matches(img1,img2,matches,corners1,corners2):
# 	kp1 = []
# 	kp2 = []
# 	for match in matches:
# 		corner1_ind = match[0]
# 		corner2_ind = match[1]
# 		kp1.append([corners1[corner1_ind,0],corners1[corner1_ind,1]])
# 		kp2.append([corners2[corner2_ind,0],corners2[corner2_ind,1]])


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

	corners1 = np.array([[20,20],[60,60],[150,150]])
	corners2 = np.array([[20,20],[60,60],[150,150]])

	img1 = cv2.imread('../Data/Train/Set1/2.jpg',cv2.IMREAD_GRAYSCALE)
	featureVec1 = get_feature_vectors(img1,corners1)
	# print(featureVec1)

	img2 = cv2.imread('../Data/Train/Set1/1.jpg',cv2.IMREAD_GRAYSCALE)
	featureVec2 = get_feature_vectors(img2,corners2)
	# print(featureVec2)

	"""
	Feature Matching
	Save Feature Matching output as matching.png

	"""

	# matches = match_features(featureVec1,featureVec2)
	matches = [[0,0],[1,1]]
	matchesImg = []

	#create ketypoints
	# corner1_keypoints = []
	# for cornerInd in range(corners1.shape[0]):
	# 	corner1_keypoints.append(cv2.KeyPoint(corners1[cornerInd,0], corners1[cornerInd,1], 5))

	# corner2_keypoints = []
	# for cornerInd in range(corners2.shape[0]):
	# 	corner2_keypoints.append(cv2.KeyPoint(corners2[cornerInd,0], corners2[cornerInd,1], 5))

	# matches_keypoints = []
	# for match in matches:
	# 	matches_keypoints.append()

	# matchesImg = cv2.drawMatches(img1,corner1_keypoints,img2,corner2_keypoints,matches,matchesImg)
	# print_matches(img1,img2,matches,corners1,corners2)

	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    
if __name__ == '__main__':
    main()
 
