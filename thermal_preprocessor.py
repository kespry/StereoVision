 #!/usr/bin/env python

'''
Note: cv2 fast Nl means denoising is too slow in python 2...
As a result, this file must be run in python 3 to achieve sufficient speed.
'''

'''
Call script using the following syntax (with conda ThermalConvert environment activated):
python thermal_preprocessor.py --folder src_images_folder_path

Where:
src_images_folder_path is the full folder path to the source thermal images

Optional Flags:
--suffix: optional suffix string to add to saved processed files
--display_images: optional flag that will display images before and after processing
--do_not_save: optional flag that will cause the program not to write the processed images to file

Modified files are saved to the source images folder.
'''

import cv2, os
from argparse import ArgumentParser
import numpy as np

def rescale_tiff(nparray):
    #set bounds of rescaling based on min and max values
    #in the entire image
    arrmax = np.max(nparray)
    arrmin = np.min(nparray)

    #rescale the values for each pixel
    out = (nparray-arrmin)*255.0/(arrmax-arrmin)

    return out.astype(np.uint8)

#Command line arguments for function
parser = ArgumentParser(description = 'Preprocess thermal calibration images to increase contrast for checkerboard finding.')
parser.add_argument('--folder', help = 'Folder where unprocessed images are stored. All images in this folder will be processed.'
                                     'These are expected in a format readable by cv2.imread().'
                                     'Output processed images will be saved with same file name'
                                     'in same folder unless otherwise specified with "suffix" argument.')
parser.add_argument('--suffix', help = 'Suffix to add to file names of processed images.', default = '')
parser.add_argument('--display_images', help = 'Display the images as they are processed.', action = 'store_true')
parser.add_argument('--do_not_save', help = 'Do not save the results of image preprocessing (just a test run).', action = 'store_false')
args = parser.parse_args()

#Discover thermal tiff files to iterate over
files = os.listdir(args.folder)
new_files = []
for f in files:
    if f.split('.')[-1] in ['tiff', 'tif', 'jpg', 'jpeg', 'png']:
        new_files.append(f)
files = new_files

print('Files to be preprocessed:\n')
print('\n'.join(files))

tmp = input('\nPress any button to proceed with processing.')

for f in files:
    file = args.folder + '/' + f
    print('Reading file: {0}'.format(file))

    therm = cv2.imread(file, 0)

    #rescale images to 0-255 scale and invert
    therm = rescale_tiff(therm)
    therm = np.invert(therm)

    #display the imported image before denoising
    if args.display_images:
        #resize to fit on screen
        scaling = therm.shape[0]/800.0
        fy,fx = [int(s/scaling) for s in therm.shape[:2]]
        resized_img = cv2.resize(therm, (fx, fy), interpolation = cv2.INTER_AREA)

        title = f + ' before denoising'
        cv2.imshow(title , resized_img)
        if cv2.waitKey(0):
            cv2.destroyWindow(title)

    #denoise the image
    print('Denoising and thresholding image. This may take a while.')
    therm = cv2.fastNlMeansDenoising(therm, None, 10, 11, 71)

    #use an adaptive threshold to increase the contrast of the checkerboard
    therm = cv2.adaptiveThreshold(therm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75,2)

    if args.display_images:
        #resize to fit on screen
        resized_img = cv2.resize(therm, (fx, fy), interpolation = cv2.INTER_AREA)

        #display converted image
        title = f + ' after denoising'
        cv2.imshow(title, resized_img)
        if cv2.waitKey(0):
            cv2.destroyWindow(title)

    if args.do_not_save:
        new_file = '{0}/{1}{2}.jpg'.format(args.folder, f[:-5], args.suffix)
        print('Saved processed image as {0}'.format(new_file))
        cv2.imwrite(new_file, therm, [cv2.IMWRITE_JPEG_QUALITY, 100])
