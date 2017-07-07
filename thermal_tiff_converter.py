 #!/usr/bin/env python

'''
Note: cv2 fast Nl means denoising is too slow in python 2...
As a result, this file must be run in python 3 to achieve sufficient speed.
'''

'''
Call script using the following syntax (with conda StereoCamera environment activated):
python thermal_tiff_converter.py src_images_folder_path show_images_flag

Where:
src_images_folder_path is a string in quotes with the full folder path to the source tiffs
show_images is either True or False (True will show modified images before final save)

Modified files are saved to the source images folder.
'''

from PIL import Image
import cv2, os, sys
import numpy as np
from matplotlib import pyplot as plt

def rescale_tiff(nparray):
	#set bounds of rescaling based on min and max values
	#in the entire image
	arrmax = np.max(nparray)
	arrmin = np.min(nparray)

	#rescale the values for each pixel
	out = (nparray-arrmin)*255.0/(arrmax-arrmin)

	return out.astype(np.uint8)

show_images = True
start_directory = os.getcwd()

#read the destination/source folders from command line
#and set show_images flag
varargin = sys.argv[1:]
nargs = len(sys.argv)
if nargs < 1:
	raise RuntimeError('Required location of source image folder was not supplied.')
elif nargs > 1:
	show_images = varargin[1]

folder = varargin[0]

#Discover tiff files and save as grayscale jpgs.
files = os.listdir(folder)
new_files = []
for f in files:
	if f[-5:] == '.tiff':
		new_files.append(f)
files = new_files

jpg_files = []

for f in files:
	if f[-5:] == '.tiff':
		file = folder + '/' + f
		print('Converting file: {}'.format(file))
		therm_tiff = Image.open(file)
		
		#convert to numpy array with same dimensions
		therm_tiff = np.array(therm_tiff, dtype = np.float_)

		
		#rescale images to 0-255 scale and invert
		therm_array = rescale_tiff(therm_tiff)
		therm_array = np.invert(therm_array)

		#new image file name
		new_file = '{0}/{1}.jpg'.format(folder, f[:-5])
		#save image
		plt.imsave(fname = new_file, arr = therm_array)

		jpg_files.append(new_file)

#convert size of new jpgs to larger format
print('Resizing jpgs by 900%.')

#change directory for magick to work
os.chdir(folder)

#resize the jpg in place
os.system('magick mogrify -resize 900% *.jpg')

#move back to start directory
os.chdir(start_directory)

print('Resizing complete.')

for f in jpg_files:
	print('Reading ', f)
	#open jpg
	img = cv2.imread(f, 0)

	if show_images:
		#resize to fit on screen
		scaling = img.shape[0]/800.0
		fy,fx = [int(s/scaling) for s in img.shape[:2]]
		resized_img = cv2.resize(img, (fx, fy), interpolation = cv2.INTER_AREA)

		title = f + ' before denoising'
		cv2.imshow(title , resized_img)
		if cv2.waitKey(0):
			cv2.destroyWindow(f)

	#denoise the image
	print('Denoising image. This may take a while.')
	img = cv2.fastNlMeansDenoising(img, None, 10, 11, 71)

	#use an adaptive threshold to increase the contrast of the checkerboard
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75,2)

	if show_images:
		#resize to fit on screen
		resized_img = cv2.resize(img, (fx, fy), interpolation = cv2.INTER_AREA)

		#display converted image
		cv2.imshow(f, resized_img)
		if cv2.waitKey(0):
			cv2.destroyWindow(f)

	cv2.imwrite(f, img)