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
from random import randrange

def random_crop(img):
	margin_x = 40
	margin_y = 40

	m_x = 128
	m_y = 128

	start_ind_x = img.shape[1]/2
	start_ind_y = img.shape[0]/2

	rand_x = randrange(margin_x,img.shape[1]-margin_x-m_x)
	rand_y = randrange(margin_y,img.shape[0]-margin_y-m_y)

	corners = [[rand_y,rand_x]]
	corners.append([rand_y+m_y,rand_x])
	corners.append([rand_y+m_y,rand_x+m_x])
	corners.append([rand_y,rand_x+m_x])
	print(img.shape)
	cropped_img = img[rand_y:rand_y+m_y,rand_x:rand_x+m_x]

	return cropped_img,corners

def get_transformed_image(img,corners):
	start = corners[0]
	end = corners[2]
	print(start,end)
	cropped_img = img[start[0]:end[0],start[1]:end[1]]
	print(cropped_img.shape)
	return cropped_img

def gen_perturbed_corners(corners):
	perturbation = 32
	new_corners = []
	for corner in corners:
		rand_x = randrange(0,perturbation)
		rand_y = randrange(0,perturbation)
		new_corners.append([corner[0]+rand_x,corner[1]+rand_y])
	return new_corners

def main():
	img = cv2.imread('../Data/Train/1.jpg')
	img = cv2.resize(img,(320,240))
	cropped_img,intial_corners = random_crop(img)
	# print('initial corners = ',intial_corners)
	perturbed_corners = gen_perturbed_corners(intial_corners)
   	# print('perturbed corners = ',perturbed_corners)
   	h = cv2.getPerspectiveTransform(np.array(perturbed_corners, dtype = "float32"),np.array(intial_corners, dtype = "float32"))
   	print(h)
   	# h_inv = np.linalg.inv(h)

   	warpped_img = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))

   	transformed_image = get_transformed_image(warpped_img,intial_corners)

if __name__ == '__main__':
    main()
 
