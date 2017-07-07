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

Example command for test images folder:
```bash
time calibrate_cameras --rows 6 --columns 9 --square-size 5 ../test_images/3/ ../test_images/test_set_3_results/
```
More information on running the library can be found as published by the original author on his [blog](https://erget.wordpress.com/2014/02/28/calibrating-a-stereo-pair-with-python/).

## Note on Thermal Imagery
For raw thermal image files, a special process is required to make them usable in the calibration.

First, some setup is required:
 1. Clone the environment using conda. A separate environment is required since the cv2 denoising function runs too slowly in the python 2   environment required for the camera calibration:
  ```python
  conda env create -f environment_thermal_tiff_resize.yml
  ```
 2. ```source activate ThermalConvert```
 
With the environment set, run the following command:

```bash python thermal_tiff_converter.py source_tiff_folder show_images```
 
```bash source_tiff_folder``` is the full path of the folder containing the images to be reprocessed

```bash show_images``` (optional) is a boolean that may be set to True to display each image as it is processed
 
Example command:
```bash python thermal_tiff_converter.py '~/tiff_folder/' True```
