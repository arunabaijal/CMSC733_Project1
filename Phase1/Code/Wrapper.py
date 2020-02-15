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
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import sys
import random
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
	arr1 = arr[:]
	minVal = min(arr1)[0]
	arr1.remove(minVal)
	minVal = min(arr1)[0]
	return minVal

'''
feature matching function
'''
def match_features(featureVec1,featureVec2):
	# print(featureVec1[0][0])
	# print(featureVec2[0][0])
	matches = []
	matching_ratio_threshold = 0.8
	for vec1_ind,vec1 in enumerate(featureVec1):
		sum_sq_dist = []
		for vec2 in featureVec2:
			# print(vec1)
			# print(vec2)
			# print(sum((vec1-vec2)**2))
			# quit()
			sum_sq_dist.append(sum((vec1-vec2)**2))
		min_val = min(sum_sq_dist)[0]
		# print(min_val)
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


def print_matches(newcords1,newcords2,matches,img1,img2):
	corner1_keypoints = []
	for cornerInd in range(newcords1.shape[0]):
		# print("img1 keypoints",newcords1[cornerInd])
		corner1_keypoints.append(cv2.KeyPoint(newcords1[cornerInd, 0], newcords1[cornerInd, 1], 5))

	corner2_keypoints = []
	for cornerInd in range(newcords2.shape[0]):
		# print("img2 keypoints",newcords2[cornerInd])
		corner2_keypoints.append(cv2.KeyPoint(int(newcords2[cornerInd, 0]), int(newcords2[cornerInd, 1]), 5))

	# matches_keypoints = []
	# for match in matches:
	# 	matches_keypoints.append()
	matchesImg = np.array([])
	dmatchvec = []
	for m in matches:
		# print("picture 1 2 coords", m)
		dmatchvec.append(cv2.DMatch(m[0], m[1], 1))

	# print(corner1_keypoints, corner2_keypoints)
	# print(dmatchvec)
	matchesImg = cv2.drawMatches(img1, corner1_keypoints, img2, corner2_keypoints, dmatchvec, matchesImg)


	cv2.imshow("Matches after Ransac", matchesImg)
	cv2.waitKey(0)

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
	img1 = cv2.imread("../Data/Train/Set1/1.jpg", 0)
	img2 = cv2.imread("../Data/Train/Set1/2.jpg", 0)
	img1_color = cv2.imread("../Data/Train/Set1/1.jpg", 1)
	img2_color = cv2.imread("../Data/Train/Set1/2.jpg", 1)
	img1_color2 = img1_color.copy()
	img1 = np.float32(img1)
	img2 = np.float32(img2)
	corner_img1 = cv2.cornerHarris(img1,2,3, 0.04)
	corner_img2 = cv2.cornerHarris(img2,2,3, 0.04)
	print(len(corner_img1))
	# cv2.imshow("corner_img",corner_img1)
	# cv2.waitKey(0)
	# result is dilated for marking the corners, not important
	corner_img1 = cv2.dilate(corner_img1, None)
	corner_img2 = cv2.dilate(corner_img2, None)

	# # Threshold for an optimal value, it may vary depending on the image.
	# img1_color[corner_img1 > 0.008 * corner_img1.max()] = [0, 0, 255]
	# cv2.imwrite("cornerImage.jpg", img1_color)
	newcords1 = applyANMS(corner_img1, 50)
	newcords2 = applyANMS(corner_img2, 50)

	fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
	axes.imshow(img1_color2, cmap=plt.cm.gray)
	axes.autoscale(False)
	axes.plot(newcords1[:, 1], newcords1[:, 0], 'r.')
	axes.axis('off')
	axes.set_title('1')

	fig.tight_layout()

	plt.show()

	fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
	axes.imshow(img2_color, cmap=plt.cm.gray)
	axes.autoscale(False)
	axes.plot(newcords2[:, 1], newcords2[:, 0], 'r.')
	axes.axis('off')
	axes.set_title('2')

	fig.tight_layout()

	plt.show()

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	featureVec1 = get_feature_vectors(img1, newcords1)
	featureVec2 = get_feature_vectors(img2, newcords2)

	print(len(featureVec1))
	print(len(featureVec2))
	# print(featureVec2[0])

	# img2 = cv2.imread('../Data/Train/Set1/2.jpg',cv2.IMREAD_GRAYSCALE)
	# featureVec1 = get_feature_vectors(img1,newcords)
	# print(featureVec1)p

	#img2 = cv2.imread('../Data/Train/Set1/1.jpg',cv2.IMREAD_GRAYSCALE)
	# featureVec2 = get_feature_vectors(img2,newcords)
	# print(featureVec2)

	"""
	Feature Matching
	Save Feature Matching output as matching.png

	"""

	matches = match_features(featureVec1,featureVec2)
	print("matches",matches)
	# matches = [[0,0],[1,1]]
	# matchesImg = []

	#create ketypoints
	corner1_keypoints = []
	for cornerInd in range(newcords1.shape[0]):
		corner1_keypoints.append(cv2.KeyPoint(newcords1[cornerInd,1], newcords1[cornerInd,0], 5))

	corner2_keypoints = []
	for cornerInd in range(newcords2.shape[0]):
		corner2_keypoints.append(cv2.KeyPoint(newcords2[cornerInd,1], newcords2[cornerInd,0], 5))

	# matches_keypoints = []
	# for match in matches:
	# 	matches_keypoints.append()
	matchesImg = np.array([])
	dmatchvec= []
	for m in matches:
		dmatchvec.append(cv2.DMatch(m[0], m[1], 1))

	matchesImg = cv2.drawMatches(img1_color2,corner1_keypoints,img2_color,corner2_keypoints,dmatchvec,matchesImg)
	cv2.imshow("Matches before ransac", matchesImg)
	cv2.waitKey(0)
	inliers_src, inliers_dst, matches_inliers = ransac(matches, newcords1, newcords2, img1_color2, img2_color)
	# print_matches(np.array(inliers_src), np.array(inliers_dst), matches_inliers, img1_color2, img2_color)
	H = recalculate_homography(inliers_src, inliers_dst)
	# print(H)
	# identity_matrix = np.identity(3)
	# identity_matrix[0][2] = float(H[0][2])*img1_color2.shape[0]
	# identity_matrix[1][2] = float(H[1][2])*img1_color2.shape[1]
	# H = np.matmul(identity_matrix, H)
	pts = [[0,0,img1_color2.shape[1],img1_color2.shape[1]],[0,img1_color2.shape[0],0,img1_color2.shape[0]],[1,1,1,1]]
	ptsWarped = np.matmul(H, pts)
	for i in range(4):
		ptsWarped[:,i] = ptsWarped[:,i]/ptsWarped[2][i]
	translation_x = min(ptsWarped[0,:])
	translation_y = min(ptsWarped[1,:])
	xmax = max(ptsWarped[0,:])
	ymax = max(ptsWarped[1,:])
	totaly = ymax - translation_y
	totalx = xmax - translation_x
	identity_matrix = np.identity(3)
	identity_matrix[0][2] = -translation_x
	identity_matrix[1][2] = -translation_y
	print("Warped corner points",ptsWarped)
	print("Identity matrix", identity_matrix)
	H = np.matmul(identity_matrix, H)
	# print(H)
	# print(ptsWarped)
	# print(translation_x)
	# print(translation_y)
	# img1 = np.pad(img1, ((translation_y,0),(translation_x,0)), mode = 'constant')
	# cv2.imwrite("paddedImage.png", img1)
	warped = cv2.warpPerspective(src=img1_color2, M=H, dsize=(int(totalx),int(totaly)))
	cv2.imwrite("Warped.png",warped)
	# print(img1_color2.shape)
	for i in inliers_src:
		i.append(1)
	# print(inliers_src)
	inliers_src_after_warping = np.matmul(H, np.transpose(inliers_src))
	inliers_src_after_warping = np.transpose(inliers_src_after_warping)
	inliers_src_warped = []
	for i in inliers_src_after_warping:
		inliers_src_warped.append([int(i[0]/i[2]), int(i[1]/i[2])])
	# print(inliers_src_after_warping)
	# print(inliers_src_warped)
	# print(warped.shape)

	#new match points in img1
	# print_matches(np.array(inliers_src_warped), np.array(inliers_dst), matches_inliers, warped, img2_color)
	matches_inliers = np.array(matches_inliers)
	xDest, yDest = inliers_dst[matches_inliers[0][1]]
	xSrc, ySrc = inliers_src_warped[matches_inliers[0][0]]
	# print(xDest, yDest)
	# print(xSrc, ySrc)

	xShift = xSrc - xDest
	yShift = ySrc - yDest
	print("X shift", xShift)
	print("Y shfit", yShift)
	lpad = 0
	rpad = 0
	upad = 0
	dpad = 0
	if xShift < 0:
		lpad = -xShift
		if img2_color.shape[1] > warped.shape[1] + lpad:
			rpad = img2_color.shape[1] - warped.shape[1] - lpad
	else:
		if xShift + img2_color.shape[1] > warped.shape[1] + lpad:
			rpad = xShift + img2_color.shape[1] - warped.shape[1] - lpad
	if yShift < 0:
		upad = -yShift
		if img2_color.shape[0] > warped.shape[0] + upad:
			dpad = img2_color.shape[0] - warped.shape[0] - upad
	else:
		if yShift + img2_color.shape[0] > warped.shape[0] + upad:
			dpad = yShift + img2_color.shape[0] - warped.shape[0] - upad
	print(warped.shape)
	print("All paddings", lpad, rpad, upad, dpad)
	warped = np.pad(warped, ((upad, dpad), (lpad, rpad), (0, 0)), mode='constant')
	print(warped.shape)
	xShift = max(0, xShift)
	yShift = max(0, yShift)
	for x in range(xShift, xShift + img2_color.shape[1]):
		for y in range(yShift, yShift + img2_color.shape[0]):
			img2X = x - xShift
			img2Y = y - yShift
			val = img2_color[img2Y,img2X,:]
			warped[y,x,:] = val
	# mask = 255 * np.ones(img2_color.shape, img2_color.dtype)
	# width, height, channels = img2_color.shape
	# center = ((width / 2) + xShift, (height / 2) + yShift)
	# print('center')
	# print(center)
	# center = (warped.shape[1]/2+5,warped.shape[0]/2+100)
	# print(warped.shape)
	# # warped = np.pad(warped, (200,200), mode='edge')
	# warped_mixed_clone = cv2.seamlessClone(img2_color, warped, mask, center, cv2.MIXED_CLONE)
	# cv2.imshow("Seamless clone", warped_mixed_clone)
	# cv2.waitKey(0)
	cv2.imshow("Final warped", warped)
	cv2.waitKey(0)
	img3 = cv2.imread("../Data/Train/Set1/3.jpg", 0)
	img3_color = cv2.imread("../Data/Train/Set1/3.jpg", 1)
	img3 = np.float32(img3)
	corner_img3 = cv2.cornerHarris(img3, 2, 3, 0.04)
	warpedGray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	warpedGray = np.float32(warpedGray)
	corner_img_warped = cv2.cornerHarris(warpedGray, 2, 3, 0.04)
	corner_img3 = cv2.dilate(corner_img3, None)
	corner_img_warped = cv2.dilate(corner_img_warped, None)
	newcords3 = applyANMS(corner_img3, 50)
	newcords_warped = applyANMS(corner_img_warped, 50)
	featureVec_warped = get_feature_vectors(warpedGray, newcords_warped)
	featureVec3 = get_feature_vectors(img3, newcords3)
	matches3 = match_features(featureVec_warped, featureVec3)
	print("matches", matches3)
	print("newcords_warped", newcords_warped)
	print("newcords3", newcords3)
	inliers_src3, inliers_dst3, matches_inliers3 = ransac(matches3, newcords_warped, newcords3, warped, img3_color)

	H3 = recalculate_homography(inliers_src3, inliers_dst3)
	pts = [[0, 0, warped.shape[1]-1, warped.shape[1]-1], [0, warped.shape[0]-1, 0, warped.shape[0]-1], [1, 1, 1, 1]]
	ptsWarped = np.matmul(H3, pts)
	for i in range(4):
		ptsWarped[:, i] = ptsWarped[:, i] / ptsWarped[2][i]
	translation_x = min(ptsWarped[0, :])
	translation_y = min(ptsWarped[1, :])
	xmax = max(ptsWarped[0, :])
	ymax = max(ptsWarped[1, :])
	totaly = ymax - translation_y
	totalx = xmax - translation_x
	identity_matrix = np.identity(3)
	identity_matrix[0][2] = -translation_x
	identity_matrix[1][2] = -translation_y
	print("Warped corner points", ptsWarped)
	print("Identity matrix", identity_matrix)
	H3 = np.matmul(identity_matrix, H3)
	warped3 = cv2.warpPerspective(src=warped, M=H3, dsize=(int(totalx), int(totaly)))
	cv2.imwrite("Warped2.png", warped3)
	for i in inliers_src3:
		i.append(1)
	# print(inliers_src)
	inliers_src_after_warping = np.matmul(H3, np.transpose(inliers_src3))
	inliers_src_after_warping = np.transpose(inliers_src_after_warping)
	inliers_src_warped = []
	for i in inliers_src_after_warping:
		inliers_src_warped.append([int(i[0]/i[2]), int(i[1]/i[2])])

	matches_inliers3 = np.array(matches_inliers3)
	xDest, yDest = inliers_dst3[matches_inliers3[0][1]]
	xSrc, ySrc = inliers_src_warped[matches_inliers3[0][0]]
	# print(xDest, yDest)
	# print(xSrc, ySrc)

	xShift = xSrc - xDest
	yShift = ySrc - yDest
	print("X shift", xShift)
	print("Y shfit", yShift)
	lpad = 0
	rpad = 0
	upad = 0
	dpad = 0
	if xShift < 0:
		lpad = -xShift
		if img3_color.shape[1] > warped3.shape[1] + lpad:
			rpad = img3_color.shape[1] - warped3.shape[1] - lpad
	else:
		if xShift + img3_color.shape[1] > warped3.shape[1] + lpad:
			rpad = xShift + img3_color.shape[1] - warped3.shape[1] - lpad
	if yShift < 0:
		upad = -yShift
		if img3_color.shape[0] > warped3.shape[0] + upad:
			dpad = img3_color.shape[0] - warped3.shape[0] - upad
	else:
		if yShift + img3_color.shape[0] > warped3.shape[0] + upad:
			dpad = yShift + img3_color.shape[0] - warped3.shape[0] - upad
	print(warped3.shape)
	print("All paddings", lpad, rpad, upad, dpad)
	warped3 = np.pad(warped3, ((upad, dpad), (lpad, rpad), (0, 0)), mode='constant')
	print(warped3.shape)
	xShift = max(0, xShift)
	yShift = max(0, yShift)
	for x in range(xShift, xShift + img3_color.shape[1]):
		for y in range(yShift, yShift + img3_color.shape[0]):
			img2X = x - xShift
			img2Y = y - yShift
			val = img3_color[img2Y, img2X, :]
			warped3[y, x, :] = val
	cv2.imshow("warped_final_3", warped3)
	cv2.waitKey(0)
	# fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
	# axes.imshow(img3_color, cmap=plt.cm.gray)
	# axes.autoscale(False)
	# axes.plot(newcords3[:, 1], newcords3[:, 0], 'r.')
	# axes.axis('off')
	# axes.set_title('Peak local max newcords3')
	#
	# fig.tight_layout()
	#
	# plt.show()
	#


	# fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
	# axes.imshow(warped, cmap=plt.cm.gray)
	# axes.autoscale(False)
	# axes.plot(newcords_warped[:, 1], newcords_warped[:, 0], 'r.')
	# axes.axis('off')
	# axes.set_title('Peak local max newcords_warped')
	#
	# fig.tight_layout()
	#
	# plt.show()
	#
	#
	# featureVec_warped = get_feature_vectors(warpedGray, newcords1)
	# featureVec3 = get_feature_vectors(img3, newcords3)
	# matches3 = match_features(featureVec_warped, featureVec3)
	# print("matches")
	# print(matches3)
	# # print("source cords", src_cords)
	# # print("destination cords", dst_cords)
	# # print_matches(np.array(newcords_warped), np.array(newcords3), matches3, warped, img3_color)
	# # print("matches")
	# # print(matches3)
	# #
	# # inliers_src3, inliers_dst3, matches_inliers3 = ransac(matches3, newcords_warped, newcords3, warped, img3_color)
	# # H3 = recalculate_homography(inliers_src3, inliers_dst3)
	# inliers_src, inliers_dst, matches_inliers = ransac(matches3, newcords_warped, newcords3, warped, img3_color)
	# print_matches(np.array(inliers_src), np.array(inliers_dst), matches_inliers, warped, img3_color)


	# print(H)
	# identity_matrix = np.identity(3)
	# identity_matrix[0][2] = float(H[0][2])*img1_color2.shape[0]
	# identity_matrix[1][2] = float(H[1][2])*img1_color2.shape[1]
	# H = np.matmul(identity_matrix, H)


	# r = warpedGray.shape[0]
	# c = warpedGray.shape[1]
	# pts3 = [[0, 0, r-1, r-1], [0, c-1, 0, c-1], [1, 1, 1, 1]]
	# ptsWarped3 = np.matmul(H3, pts3)
	# for i in range(4):
	# 	ptsWarped3[:, i] = ptsWarped3[:, i] / ptsWarped3[2][i]
	# translation_x = min(ptsWarped3[0, :])
	# translation_y = min(ptsWarped3[1, :])
	# xmax = max(ptsWarped3[0, :])
	# ymax = max(ptsWarped3[1, :])
	# totaly = ymax - translation_y
	# totalx = xmax - translation_x
	# identity_matrix = np.identity(3)
	# identity_matrix[0][2] = -translation_x
	# identity_matrix[1][2] = -translation_y
	# H3 = np.matmul(identity_matrix, H3)


	# print(H)
	# print(ptsWarped)
	# print(translation_x)
	# print(translation_y)
	# img1 = np.pad(img1, ((translation_y,0),(translation_x,0)), mode = 'constant')
	# cv2.imwrite("paddedImage.png", img1)


	# warped3 = cv2.warpPerspective(src=warped, M=H3, dsize=(int(totalx), int(totaly)))
	# cv2.imwrite("Warped3.png", warped3)
	# # print(img1_color2.shape)
	# for i in inliers_src3:
	# 	i.append(1)
	# # print(inliers_src)
	# inliers_src_after_warping3 = np.matmul(H3, np.transpose(inliers_src3))
	# inliers_src_warped3 = []
	# for i in np.transpose(inliers_src_after_warping3):
	# 	inliers_src_warped3.append([int(i[0] / i[2]), int(i[1] / i[2])])


	# print(inliers_src_after_warping)
	# print(inliers_src_warped)
	# print(warped.shape)

	# new match points in img1
	# print_matches(np.array(inliers_src_warped3), np.array(inliers_dst3), matches_inliers3, warped, img3_color)
	# matches_inliers3 = np.array(matches_inliers3)
	# xDest, yDest = inliers_dst3[matches_inliers3[0][1]]
	# xSrc, ySrc = inliers_src_warped3[matches_inliers3[0][0]]
	# print(xDest, yDest)
	# print(xSrc, ySrc)

	# xShift = xSrc - xDest
	# yShift = ySrc - yDest
	# warped3 = np.pad(warped3, ((0, xShift), (0, yShift), (0, 0)), mode='constant')
	# # print(warped3.shape, img2_color.shape)
	# # print(img2_color.shape[0] + xShift, img2_color.shape[1] + yShift)
	# for x in range(xShift, img3_color.shape[0] + xShift):
	# 	for y in range(yShift, img3_color.shape[1] + yShift):
	# 		img2X = x - xShift
	# 		img2Y = y - yShift
	# 		val = img3_color[img2X, img2Y, :]
	# 		warped3[x, y, :] = val
	# cv2.imshow("final", warped3)
	# cv2.waitKey(0)



	# oX = int(abs(translation_y))
	# oY = int(abs(translation_x))
	# print(img2_color.shape[1], oX)
	# for y in range(oY, img2_color.shape[0] + oY):
	# 	for x in range(oX, img2_color.shape[1] + oX):
	# 		img2_y = y - oY
	# 		img2_x = x - oX
	# 		warped[y, x, :] = img2_color[img2_y, img2_x, :]
	#
	# cv2.imshow("final", warped)
	# cv2.waitKey(0)
	# print(np.array(inliers_src_warped).shape)
	# print(np.array(inliers_dst).shape)
	# print(np.array(inliers_src_warped).shape, np.array(inliers_dst).shape, matches_inliers.__len__())
	# print_matches(np.array(inliers_src_warped), np.array(inliers_dst), matches_inliers, warped, img2_color)
	# print(img1_color2[0][0])
	# im_dst = cv2.warpPerspective(img1_color2, H, (2000,2000))

	# plt.imshow(warped)
	# plt.show()
	# plt.imshow(img2_color)
	# plt.show()
	# cv2.imshow("warped", warped)
	# cv2.imshow("img1", img1_color2)
	# cv2.waitKey(0)
	# print_matches(img1,img2,matches,corners1,corners2)
	# [[-4.21677918e-03 - 8.75286479e-05  9.91282038e-01]
	#  [-2.61117693e-04 - 4.15600254e-03  1.31567863e-01]
	# [-8.17347495e-07 - 5.02766641e-08 - 3.83257265e-03]]


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
def recalculate_homography(inliers_src, inliers_dst):
	A= []
	for i in range(len(inliers_src)):
		A.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0 ,0, inliers_src[i][0]*inliers_dst[i][0], inliers_src[i][1]*inliers_dst[i][0], inliers_dst[i][0]])
		A.append([ 0, 0 ,0,-inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0]*inliers_dst[i][1], inliers_src[i][1]*inliers_dst[i][1], inliers_dst[i][1]])
	s, v, vh = np.linalg.svd(A)
	H = vh[-1,:]
	H = H.reshape((3,3))
	return  H

def applyANMS(img1, nbest):
	#lm = sc.maximum_filter(cimg, size=20)
	#mask = (cimg == lm)
	# image_max = sc.maximum_filter(img1, size=20, mode='constant')

	# Comparison between image_max and im to find the coordinates of local maxima
	coordinates = peak_local_max(img1, min_distance=5)
	Nstrong = len(coordinates)
	# print(Nstrong)
	rcord = []
	for i in range(Nstrong):
		rcord.append([sys.maxsize,[coordinates[i][0],coordinates[i][1]]])
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
	# print(rcord)
	result = []
	for r in rcord:
		result.append(r[1])
	return np.asarray(result)

def ransac(matches, newcords1, newcords2, img1, img2):

	dist_thresh = 10
	n_matches_thresh = 0.46
	nMatches = len(matches)
	counter = 0
	while (counter<200):
		feature_pairs = random.sample(matches, k=4)
		src_cords = []
		dst_cords = []
		for i in feature_pairs:
			x1, y1 = newcords1[i[0]]
			x2, y2 = newcords2[i[1]]
			src_cords.append([y1, x1])
			dst_cords.append([y2, x2])
		src_cords = np.array(src_cords, dtype = "float32")
		dst_cords = np.array(dst_cords, dtype = "float32")



		# src_cords
		h = cv2.getPerspectiveTransform(src_cords, dst_cords)


		inliers_src = []
		inliers_dst = []
		matches_inliers = []
		n_inliers = 0
		for match in matches:
			p1 = np.array([newcords1[match[0]][1],newcords1[match[0]][0],1])
			# p2 = np.array([dst_cords[ind][0],dst_cords[ind][1]])

			hp1 = np.matmul(h,p1)
			hp1 = hp1/hp1[2]
			dist = (hp1[0]-newcords2[match[1]][1])**2+(hp1[1]-newcords2[match[1]][0])**2
			# print(dist)
			if dist<dist_thresh:
				# print("Matched", dist)
				matches_inliers.append([n_inliers,n_inliers])
				n_inliers += 1
				inliers_src.append([newcords1[match[0]][1],newcords1[match[0]][0]])
				inliers_dst.append([newcords2[match[1]][1],newcords2[match[1]][0]])
		# print('\n')
		print(float(len(inliers_src))/len(matches))
		if(float(len(inliers_src))/len(matches)>n_matches_thresh):
			break
		counter+=1
	if(counter==200):
		print('not found!!')
	else:
		# print_matches(np.array(inliers_src), np.array(inliers_dst), matches_inliers, img1, img2)

		return inliers_src, inliers_dst, matches_inliers
		# h_final, status = cv2.findHomography(np.array(inliers_src), np.array(inliers_dst))
		# # print(h, status)
		# im_dst = cv2.warpPerspective(img1, h_final, (img1.shape[1], img1.shape[0]))
		# cv2.imshow("new image",im_dst)
		# cv2.imshow("img2", img2)
		# cv2.imshow("img1", img1)
		# cv2.waitKey(0)

    
if __name__ == '__main__':
    main()
 
