#!/usr/bin/env python

'''Note: tested in python 3 only'''

from PIL import Image
import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imfilter

def rescale_tiff(nparray):
	arrmax = np.max(nparray)
	arrmin = np.min(nparray)
	out = (nparray-arrmin)*255.0/(arrmax-arrmin)
	return out.astype(np.uint8)

#folder = '/Users/croomjm1/version-control/StereoVisionKespry/test_images/multi-mode_calibration_6-22-17/duo-visual-thermal'
folder = '/Users/croomjm1/Downloads/duo_thermal_output_set'
#folder = input("Folder: ")

"""Discover tiff files and save as grayscale jpgs."""
files = os.listdir(folder)
new_files = []
for f in files:
	if f[-5:] == '.tiff':
		new_files.append(f)
files = new_files


for f in files:
	if f[-5:] == '.tiff':
		file = folder + '/' + f
		print('file = {}'.format(file))
		therm_tiff = Image.open(file)
		
		#convert to numpy array with same dimensions
		therm_tiff = np.array(therm_tiff, dtype = np.float_)

		therm_tiff_sharpened = imfilter(therm_tiff, 'sharpen')
		
		#rescale images to 0-255 scale and invert
		therm_array = rescale_tiff(therm_tiff)
		therm_array = np.invert(therm_array)

		#sharpen using scipy.misc.imfilter
		#therm_array_sharpened = imfilter(therm_array, 'sharpen')

		#apply threshhold to sharpened image
		#therm_array_sharpened_threshholded = np.copy(therm_array_sharpened)
		#therm_array_sharpened_threshholded[therm_array_sharpened_threshholded>115] = 255


		#display converted image
		plt.figure(f)
		plt.gray()

		#plot therm_tiff
		#plt.subplot(221)
		#plt.imshow(therm_tiff)

		#plot therm array
		#plt.subplot(222)
		#plt.imshow(therm_array)

		#plot sharpened therm_array
		#plt.subplot(223)
		#plt.imshow(therm_array_sharpened)

		#plot sharpened and threshholded therm array
		#plt.subplot(224)
		#plt.imshow(therm_array_sharpened_threshholded)

		#plt.show(block = 'False')

		#new image file name
		new_file = '{0}/{1}.jpg'.format(folder, f[:-5])
		#save image
		plt.imsave(fname = new_file, arr = therm_array)

		#open newly created file and modify it
		img = cv2.imread(new_file, 0)
		#gaussian_blurred_img = cv2.GaussianBlur(img, (9,9), 10.0)
		#unsharp_img = cv2.addWeighted(img, 1.5, gaussian_blurred_img, -0.5, 0, img)
		denoise_img = cv2.fastNlMeansDenoising(img, None, 10, 11, 71)
		adaptive_thresh_img = cv2.adaptiveThreshold(denoise_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75,2)


		#cv2.imshow(f + 'Unsharp', unsharp_img)
		#cv2.imshow(f + 'Denoised', denoise_img)
		#cv2.imshow(f + 'Adaptive Thresh', adaptive_thresh_img)

		cv2.imwrite(new_file, adaptive_thresh_img)
		#if cv2.waitKey(0):
		#	cv2.destroyWindow(f)