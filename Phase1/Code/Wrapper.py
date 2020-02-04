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
import scipy.ndimage.filters as sc
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import sys
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
	img1 = cv2.imread("/home/aruna/abaijal_p1/Phase1/Data/Train/Set1/1.jpg", 0)
	img1_color = cv2.imread("/home/aruna/abaijal_p1/Phase1/Data/Train/Set1/1.jpg", 1)
	img1_color2 = img1_color.copy()
	img1 = np.float32(img1)
	corner_img1 = cv2.cornerHarris(img1,2,3, 0.04)
	print(len(corner_img1))
	# cv2.imshow("corner_img",corner_img1)
	# cv2.waitKey(0)
	# result is dilated for marking the corners, not important
	corner_img1 = cv2.dilate(corner_img1, None)

	# Threshold for an optimal value, it may vary depending on the image.
	img1_color[corner_img1 > 0.008 * corner_img1.max()] = [0, 0, 255]
	cv2.imwrite("cornerImage.jpg", img1_color)
	newcords = applyANMS(corner_img1, 50)
	# fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
	# axes.imshow(img1_color2, cmap=plt.cm.gray)
	# axes.autoscale(False)
	# axes.plot(newcords[:, 1], newcords[:, 0], 'r.')
	# axes.axis('off')
	# axes.set_title('Peak local max')
	#
	# fig.tight_layout()
	#
	# plt.show()

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	featureVec2 = get_feature_vectors(img1, newcords)

	# img2 = cv2.imread('../Data/Train/Set1/2.jpg',cv2.IMREAD_GRAYSCALE)
	# featureVec1 = get_feature_vectors(img1,newcords)
	# print(featureVec1)

	#img2 = cv2.imread('../Data/Train/Set1/1.jpg',cv2.IMREAD_GRAYSCALE)
	# featureVec2 = get_feature_vectors(img2,newcords)
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
	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
def applyANMS(img1, nbest):
	#lm = sc.maximum_filter(cimg, size=20)
	#mask = (cimg == lm)
	# image_max = sc.maximum_filter(img1, size=20, mode='constant')

	# Comparison between image_max and im to find the coordinates of local maxima
	coordinates = peak_local_max(img1, min_distance=5)
	Nstrong = len(coordinates)
	print(Nstrong)
	rcord = []
	for i in range(Nstrong):
		rcord.append([sys.maxsize,[coordinates[i][0],coordinates[i][1]]])
	eucDist = sys.maxsize

	for i in range(Nstrong-1):
		for j in range(i+1,Nstrong):
			xi = coordinates[i][0]
			yi = coordinates[i][1]
			xj = coordinates[j][0]
			yj = coordinates[j][1]
			# print(xj, yj, xi, yi)
			#if img1[xj][yj] > img1[xi][yi]:
			eucDist = (xj-xi)**2 + (yj-yi)**2
			if eucDist < rcord[i][0]:
				rcord[i][0] = eucDist
				rcord[i][1] = [xi,yi]
	rcord.sort()
	rcord = rcord[::-1]
	rcord = rcord[:nbest]
	print(rcord)
	result = []
	for r in rcord:
		result.append(r[1])
	return np.asarray(result)
    
if __name__ == '__main__':
    main()
 
