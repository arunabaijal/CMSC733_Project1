#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s):
Aruna Baijal (abaijal@umd.edu)
M.Eng Student in Robotics,
University of Maryland, College Park

Ashwin Kuruttukulam (ashwinvk@umd.edu)
M.Eng Student in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
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
	matches = []
	matching_ratio_threshold = 0.8
	for vec1_ind,vec1 in enumerate(featureVec1):
		sum_sq_dist = []
		for vec2 in featureVec2:
			sum_sq_dist.append(sum((vec1-vec2)**2))
		min_val = min(sum_sq_dist)[0]
		second_min_val = find_second_smallest(sum_sq_dist)
		# print('matching ratio for corner ',str(vec1_ind),'is = ',(min_val/second_min_val))
		if((min_val/second_min_val)<matching_ratio_threshold):
			vec2_ind = sum_sq_dist.index(min_val)
			matches.append([vec1_ind,vec2_ind])
	return matches

def print_matches(newcords1,newcords2,matches,img1,img2, filename):
	corner1_keypoints = []
	for cornerInd in range(newcords1.shape[0]):
		corner1_keypoints.append(cv2.KeyPoint(newcords1[cornerInd, 0], newcords1[cornerInd, 1], 5))

	corner2_keypoints = []
	for cornerInd in range(newcords2.shape[0]):
		corner2_keypoints.append(cv2.KeyPoint(int(newcords2[cornerInd, 0]), int(newcords2[cornerInd, 1]), 5))

	matchesImg = np.array([])
	dmatchvec = []
	for m in matches:
		dmatchvec.append(cv2.DMatch(m[0], m[1], 1))

	matchesImg = cv2.drawMatches(img1, corner1_keypoints, img2, corner2_keypoints, dmatchvec, matchesImg)


	cv2.imwrite(filename, matchesImg)
	# cv2.waitKey(0)

def main():

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	basePath = ["../Data/Train/Set1/", "../Data/Train/Set2/", "../Data/Train/CustomSet1/", "../Data/Train/CustomSet2/", "../Data/Test/TestSet3/", "../Data/Test/TestSet4/", "../Data/Train/Set3/", "../Data/Test/TestSet2/"]
	paths = []
	for num in range(5):
		paths.append([basePath[num]+"1.jpg", basePath[num]+"2.jpg", "my_pano_set_"+str(num+1)+".png", basePath[num]+"3.jpg"])
	num = 5
	paths.append([basePath[num] + "1.jpg", basePath[num] + "2.jpg", "my_pano_set_" + str(num + 1) + ".png",
				  basePath[num] + "3.jpg", "my_pano_set_" + str(num + 1) + ".png", basePath[num] + "4.jpg", "my_pano_set_" + str(num + 1) + ".png", basePath[num] + "5.jpg"])
	num = 6
	paths.append([basePath[num] + "2.jpg", basePath[num] + "3.jpg", "my_pano_set_" + str(num + 1) + ".png",
				  basePath[num] + "4.jpg", basePath[num] + "8.jpg", basePath[num] + "7.jpg",
				  "my_pano_set_rev_" + str(num + 1) + ".png", basePath[num] + "6.jpg",
				  "my_pano_set_rev_" + str(num + 1) + ".png", basePath[num] + "5.jpg",
				  "my_pano_set_" + str(num + 1) + ".png", "my_pano_set_rev_" + str(num + 1) + ".png"])
	num = 7
	paths.append([basePath[num]+"1.jpg", basePath[num]+"2.jpg", "my_pano_set_"+str(num+1)+".png", basePath[num]+"3.jpg", "my_pano_set_"+str(num+1)+".png", basePath[num]+"4.jpg", "my_pano_set_"+str(num+1)+".png", basePath[num]+"5.jpg", "my_pano_set_"+str(num+1)+".png", basePath[num]+"6.jpg", basePath[num]+"9.jpg", basePath[num]+"8.jpg", "my_pano_set_rev_"+str(num+1)+".png", basePath[num]+"7.jpg", "my_pano_set_rev_"+str(num+1)+".png", "my_pano_set_"+str(num+1)+".png"])
	# paths.append([basePath[num]+"2.jpg", basePath[num]+"3.jpg", "my_pano_set"+str(num+1)+".png", basePath[num]+"4.jpg", basePath[num]+"8.jpg", basePath[num]+"7.jpg",  "my_pano_set_rev"+str(num+1)+".png", basePath[num]+"6.jpg", "my_pano_set_rev"+str(num+1)+".png", basePath[num]+"5.jpg", "my_pano_set"+str(num+1)+".png", "my_pano_set_rev"+str(num+1)+".png"])
	print(paths)
	# path = paths[4]
	for num, path in enumerate(paths):
		for j in range(0,len(path),2):
			print('Running for images ' + path[j] + ' and ' + path[j+1] + '...')
			img1_color = cv2.imread(path[j], 1)
			if j == 0 or (num == 6 and j == 4) or (num == 7 and j == 10):
				scale_percent = 70  # percent of original size
				width = int(img1_color.shape[1] * scale_percent / 100)
				height = int(img1_color.shape[0] * scale_percent / 100)
				dim = (width, height)
				# resize image
				img1_color = cv2.resize(img1_color, dim, interpolation=cv2.INTER_AREA)
			img2_color = cv2.imread(path[j+1], 1)
			if not (num == 6 and j== 10) or (num == 7 and j== 14):
				scale_percent = 70  # percent of original size
				width = int(img2_color.shape[1] * scale_percent / 100)
				height = int(img2_color.shape[0] * scale_percent / 100)
				dim = (width, height)
				# resize image
				img2_color = cv2.resize(img2_color, dim, interpolation=cv2.INTER_AREA)
			img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
			img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
			img1_color2 = img1_color.copy()
			img1 = np.float32(img1)
			img2 = np.float32(img2)
			newcords1 = applyANMS(img1, 200)
			newcords2 = applyANMS(img2, 200)

			fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
			axes.imshow(img1_color2, cmap=plt.cm.gray)
			axes.autoscale(False)
			axes.plot(newcords1[:, 1], newcords1[:, 0], 'r.')
			axes.axis('off')
			axes.set_title('ANMS')

			fig.tight_layout()

			plt.savefig('anms_'+str(j)+'_set_'+str(num)+'.png')
			plt.close()

			fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
			axes.imshow(img2_color, cmap=plt.cm.gray)
			axes.autoscale(False)
			axes.plot(newcords2[:, 1], newcords2[:, 0], 'r.')
			axes.axis('off')
			axes.set_title('ANMS')

			fig.tight_layout()

			plt.savefig('anms_'+str(j+1)+'_set_' + str(num) + '.png')
			plt.close()

			"""
			Feature Descriptors
			Save Feature Descriptor output as FD.png
			"""

			featureVec1 = get_feature_vectors(img1, newcords1)
			featureVec2 = get_feature_vectors(img2, newcords2)
			"""
			Feature Matching
			Save Feature Matching output as matching.png
		
			"""

			matches = match_features(featureVec1,featureVec2)
			corner1_keypoints = []
			for cornerInd in range(newcords1.shape[0]):
				corner1_keypoints.append(cv2.KeyPoint(newcords1[cornerInd,1], newcords1[cornerInd,0], 5))

			corner2_keypoints = []
			for cornerInd in range(newcords2.shape[0]):
				corner2_keypoints.append(cv2.KeyPoint(newcords2[cornerInd,1], newcords2[cornerInd,0], 5))
			matchesImg = np.array([])
			dmatchvec= []
			for m in matches:
				dmatchvec.append(cv2.DMatch(m[0], m[1], 1))

			matchesImg = cv2.drawMatches(img1_color2,corner1_keypoints,img2_color,corner2_keypoints,dmatchvec,matchesImg)
			cv2.imwrite("Matches_before_ransac_"+str(j) + str(j+1)+"_set_"+str(num+1)+".png", matchesImg)
			if (num == 7 and j >= 10):
				ran = ransac(matches, newcords1, newcords2, 90, 0.17, 2000)
				if ran:
					inliers_src, inliers_dst, matches_inliers = ran
				else:
					continue
			else:
				ran = ransac(matches, newcords1, newcords2, 70, 0.45, 2000)
				if ran:
					inliers_src, inliers_dst, matches_inliers = ran
				else:
					continue
			print_matches(np.array(inliers_src), np.array(inliers_dst), matches_inliers, img1_color2, img2_color, "Matches_after_ransac_"+str(j) + str(j+1)+"_set_"+str(num+1)+".png")
			H = recalculate_homography(inliers_src, inliers_dst)
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
			identity_matrix[0][2] = -1*translation_x
			identity_matrix[1][2] = -1*translation_y
			H = np.matmul(identity_matrix, H)
			warped = cv2.warpPerspective(src=img1_color2, M=H, dsize=(int(totalx),int(totaly)))
			for i in inliers_src:
				i.append(1)
			inliers_src_after_warping = np.matmul(H, np.transpose(inliers_src))
			inliers_src_after_warping = np.transpose(inliers_src_after_warping)
			inliers_src_warped = []
			for i in inliers_src_after_warping:
				inliers_src_warped.append([int(i[0]/i[2]), int(i[1]/i[2])])
			matches_inliers = np.array(matches_inliers)
			xDest, yDest = inliers_dst[matches_inliers[0][1]]
			xSrc, ySrc = inliers_src_warped[matches_inliers[0][0]]

			xShift = xSrc - xDest
			yShift = ySrc - yDest
			lpad = 0
			rpad = 0
			upad = 0
			dpad = 0
			if xShift < 0:
				lpad = -xShift
				if img2_color.shape[1] > warped.shape[1] + lpad:
					rpad = img2_color.shape[1] - warped.shape[1] - lpad
			else:
				if xShift + img2_color.shape[1] > warped.shape[1]:
					rpad = xShift + img2_color.shape[1] - warped.shape[1]
			if yShift < 0:
				upad = -yShift
				if img2_color.shape[0] > warped.shape[0] + upad:
					dpad = img2_color.shape[0] - warped.shape[0] - upad
			else:
				if yShift + img2_color.shape[0] > warped.shape[0]:
					dpad = yShift + img2_color.shape[0] - warped.shape[0]
			# print('warped shape before padding',warped.shape)
			# print('img2 shape ',img2_color.shape)
			# print('Padding required', upad, dpad, lpad, rpad)
			warped = np.pad(warped, ((upad, dpad), (lpad, rpad), (0, 0)), mode='constant')
			# cv2.imshow('warped', warped)
			# cv2.waitKey(0)
			# print('warped shape after padding', warped.shape)
			xShift = max(0, xShift)
			yShift = max(0, yShift)
			for x in range(xShift, xShift + img2_color.shape[1]):
				for y in range(yShift, yShift + img2_color.shape[0]):
					img2X = x - xShift
					img2Y = y - yShift
					val = img2_color[img2Y,img2X,:]
					warped[y,x,:] = val
			if (num == 7 and j >= 8) or (num == 6 and j >= 4):
				cv2.imwrite("my_pano_set_rev_" + str(num + 1) + ".png", warped)
			else:
				cv2.imwrite("my_pano_set_" + str(num + 1) + ".png", warped)
			cv2.imwrite("my_pano_set_" + str(num + 1) + "image" + str(j) + ".png", warped)

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
	coordinates = cv2.goodFeaturesToTrack(img1, 3*nbest, 0.05, 20)
	Nstrong = len(coordinates)
	# print(Nstrong)
	rcord = []
	eucDist = 0
	for i in range(Nstrong):
		rcord.append([sys.maxsize,[coordinates[i][0][0],coordinates[i][0][1]]])
	for i in range(Nstrong):
		for j in range(Nstrong):
			yi = int(coordinates[i][0][0])
			xi = int(coordinates[i][0][1])
			yj = int(coordinates[j][0][0])
			xj = int(coordinates[j][0][1])
			if img1[xj][yj] > img1[xi][yi]:
				eucDist = (xj-xi)**2 + (yj-yi)**2
			if eucDist < rcord[i][0]:
				rcord[i][0] = eucDist
				rcord[i][1] = [xi,yi]
	rcord.sort()
	rcord = rcord[::-1]
	rcord = rcord[:nbest]
	result = []
	for r in rcord:
		result.append(r[1])
	return np.asarray(result)

def ransac(matches, newcords1, newcords2, dist_thresh, n_matches_thresh, counter_thresh):

	dist_thresh = dist_thresh
	n_matches_thresh = n_matches_thresh
	nMatches = len(matches)
	if nMatches < 8:
		return False
	counter = 0
	while (counter<counter_thresh):
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

		h = cv2.getPerspectiveTransform(src_cords, dst_cords)

		inliers_src = []
		inliers_dst = []
		matches_inliers = []
		n_inliers = 0
		for match in matches:
			p1 = np.array([newcords1[match[0]][1],newcords1[match[0]][0],1])
			hp1 = np.matmul(h,p1)
			hp1 = hp1/hp1[2]
			dist = (hp1[0]-newcords2[match[1]][1])**2+(hp1[1]-newcords2[match[1]][0])**2
			if dist<dist_thresh:
				matches_inliers.append([n_inliers,n_inliers])
				n_inliers += 1
				inliers_src.append([newcords1[match[0]][1],newcords1[match[0]][0]])
				inliers_dst.append([newcords2[match[1]][1],newcords2[match[1]][0]])
		# print('RANSAC matches detected ' + str(float(len(inliers_src))/len(matches)))
		if(float(len(inliers_src))/len(matches)>n_matches_thresh):
			break
		counter+=1
	if(counter==counter_thresh):
		print('Not enough matches between images!')
		return False
	else:
		return inliers_src, inliers_dst, matches_inliers


if __name__ == '__main__':
    main()

