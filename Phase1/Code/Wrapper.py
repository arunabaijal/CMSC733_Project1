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
	# cv2.imshow("corner_img",corner_img1)
	# cv2.waitKey(0)
	# result is dilated for marking the corners, not important
	corner_img1 = cv2.dilate(corner_img1, None)

	# Threshold for an optimal value, it may vary depending on the image.
	img1_color[corner_img1 > 0.008 * corner_img1.max()] = [0, 0, 255]
	cv2.imwrite("cornerImage.jpg", img1_color)
	newcords = applyANMS(corner_img1, 50)
	fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
	axes.imshow(img1_color2, cmap=plt.cm.gray)
	axes.autoscale(False)
	axes.plot(newcords[:, 1], newcords[:, 0], 'r.')
	axes.axis('off')
	axes.set_title('Peak local max')

	fig.tight_layout()

	plt.show()
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

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

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
 
