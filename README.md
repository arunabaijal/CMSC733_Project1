# CMSC733_Project1

## Phase 1
To run Phase 1, you need to provide path of data sets to the variable basePath at line 118 of Wrapper.py.<br/>
In the list path, append the images in the set. Please follow order, 'img1', 'img2', 'my_pano_set_setnum', 'img3'.<br/>
That is, append the image name my_pano_set_'set_number'.png before every image starting from the 3rd image.

run on cammand line  
`python Wrapper.py`
This will generate the final panoramic image, intermediate panoramic images, matches between images before and after RANSAC and the ANMS corner detection for each image.

## Phase 2
To run Phase 2 for training, make sure Data exists in folder Data/Train.
Then on command line run
`python Train.py`

For testing, make sure Data exists in folder Data/Test and the checkpoints exist.
Then run
`python Test.py`
