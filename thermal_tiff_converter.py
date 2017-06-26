#!/usr/bin/env python

'''Note: tested in python 3 only'''

from PIL import Image
import cv2, os
import numpy as np
from matplotlib import pyplot as plt

def rescale_tiff(nparray):
	arrmax = np.max(nparray)
	arrmin = np.min(nparray)
	out = (nparray-arrmin)*255.0/(arrmax-arrmin)
	return out.astype(np.uint8)

folder = '/Users/croomjm1/version-control/StereoVisionKespry/test_images/multi-mode_calibration_6-22-17/duo-visual-thermal'
#folder = input("Folder: ")

"""Discover tiff files and save as grayscale jpgs."""
files = os.listdir(folder)

for f in files:
	if f[-5:] == '.tiff':
		file = folder + '/' + f
		print('file = {}'.format(file))
		therm_tiff = Image.open(file)
		
		#convert to numpy array with same dimensions
		therm_tiff = np.array(therm_tiff, dtype = np.float_)
		#rescale images to 0-255 scale
		therm_array = rescale_tiff(therm_tiff)

		#display converted image
		#plt.figure(f)
		plt.gray()
		#plt.imshow(therm_tiff)
		#plt.show()

		#new image file name
		new_file = '{0}/{1}.jpg'.format(folder, f[:-5])
		plt.imsave(fname = new_file, arr = therm_tiff)

		#open newly created file to check it was created correctly
		img = cv2.imread(new_file)
		cv2.imshow(f, img)
		if cv2.waitKey(0):
			cv2.destroyWindow(f)