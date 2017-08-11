## Overview
This library is forked from the [StereoVision](https://github.com/erget/StereoVision) library publicly available on GitHub. It's purpose is to find the intrinsic and extrinsic camera parameters between stereo cameras. Note that rectification is only possible for cameras with the same image size (i.e. two identical cameras). The output of the primary script is a set of numpy (.npy) files with the extrinsic and intrinsic camera parameters to be used in various computer vision applications. This output is currently used for the [Multimode Stitching](https://github.com/kespry/MultimodeStitching) repository.

## Installation
To install, follow these steps:

 1. Clone the repository.
 2. Clone the environment using conda:
 	```python
 	conda env create -f environment.yml
 	```
 3. ```source activate StereoCamera```
 4. ```python setup.py build```
 5. ```python setup.py install```

### Checking Your Setup
I've included a folder with test images to make sure everything is working correctly in ```/test_images```. I got the test images from [here](https://github.com/sourishg/stereo-calibration/tree/master/calib_imgs).

## Running the Calibration
To calibrate a stereo camera setup, images must be saved in the format _left_abc123.jpg and _right_abc123.jpg, where abc123 is any alpha-numeric suffix (that can be fed into python's ```sort()``` function). Each pair of images (i.e. right and left image of the same scene) must have the same suffix. All images to be used for calibration should be saved in the same folder without any other contents.

To run a stereo camera calibration after the library is installed, use the command

```bash
time calibrate_cameras --rows nrows --columns ncols --square-size squareSize --show-chessboards inputImageFolder outputResultsFolder
```

Where:
 * nrows, ncols = number of interior rows and columns (don't count the outermost squares on the grid)
 * squareSize = size of square in centimeters
 * --show-chessboards = (optional) displays detected pattern corners for each image (activate window and press any key to advance to next image)
 * --validate-results = (optional) uses the calculated homography matrix to project detected chessboard corners from the right frame to the left and displays all right images overlaid on the corresponding left images

Example command for test images folder:
```bash
time calibrate_cameras --rows 6 --columns 9 --square-size 5 --validate-results ../test_images/3/ ../test_images/test_set_3_results/
```
More information about the original library can be found as published by the original author on his [blog](https://erget.wordpress.com/2014/02/28/calibrating-a-stereo-pair-with-python/).

## Note on Thermal Imagery
For raw thermal image files, a special process is required to make them usable in the calibration.

### Size
First, the calibration (specifically, chessboard corner finding) works best if the thermal images are resized using image magick to a minimum size. The images have only been tested down to a minimum size of 960x768px. If the raw images are below this, use the following command to resize them to meet this minimum size. For example, if the images are .png files with starting size 320x256px, you would use:

```bash
magick mogrify -resize 300% *.png
```
### Preprocessing
First, some setup is required:
 1. Clone the environment using conda. A separate environment is required since the cv2 denoising function runs too slowly in the python 2   environment required for the camera calibration:
  ```python
  conda env create -f environment_thermal_tiff_resize.yml
  ```
 2. ```source activate ThermalConvert```
 
With the environment set, run the following command:

```bash
python thermal_preprocessor.py --folder source_thermal_folder --suffix processed_file_suffix --display_images --do_not_save
```
 
```source_thermal_folder``` is the full path of the folder containing the thermal images to be preprocessed. Note that all images in this folder will be processed.

```processed_file_suffix``` (optional) is a suffix to be added to the end of the file names before processing to create the processed image file names.

```--display_images``` (optional) is a flag to display each image as it is processed

```--do_not_save``` (optional) is a flag to indicate that the processed images should not be saved.
 
Example command:
```bash
python thermal_tiff_converter.py --folder ~/tiff_folder/' --display_images
```

The processed images will be saved in the same folder as the source images with the same name (and added suffix, if applicable) as .jpg files.
