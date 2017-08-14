# Copyright (C) 2014 Daniel Lee <lee.daniel.1986@gmail.com>
#
# This file is part of StereoVision.
#
# StereoVision is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StereoVision is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoVision.  If not, see <http://www.gnu.org/licenses/>.

"""
Classes for calibrating homemade stereo cameras.

Classes:

    * ``StereoCalibration`` - Calibration for stereo camera
    * ``StereoCalibrator`` - Class to calibrate stereo camera with

.. image:: classes_calibration.svg
"""

import os, cv2
from PIL import Image
from progressbar import ProgressBar, Percentage, Bar
import numpy as np
from stereovision.exceptions import ChessboardNotFoundError


class StereoCalibration(object):

    """
    A stereo camera calibration.

    The ``StereoCalibration`` stores the calibration for a stereo pair. It can
    also rectify pictures taken from its stereo pair.
    """

    def __str__(self):
        output = ""
        for key, item in self.__dict__.items():
            output += key + ":\n"
            output += str(item) + "\n"
        return output

    def _copy_calibration(self, calibration):
        """Copy another ``StereoCalibration`` object's values."""
        for key, item in calibration.__dict__.items():
            self.__dict__[key] = item

    def _interact_with_folder(self, output_folder, action):
        """
        Export/import matrices as *.npy files to/from an output folder.

        ``action`` is a string. It determines whether the method reads or writes
        to disk. It must have one of the following values: ('r', 'w').
        """
        if not action in ('r', 'w'):
            raise ValueError("action must be either 'r' or 'w'.")
        for key, item in self.__dict__.items():
            if isinstance(item, dict):
                for side in ("left", "right"):
                    filename = os.path.join(output_folder,
                                            "{}_{}.npy".format(key, side))
                    if action == 'w':
                        np.save(filename, self.__dict__[key][side])
                    else:
                        self.__dict__[key][side] = np.load(filename)
            else:
                filename = os.path.join(output_folder, "{}.npy".format(key))
                if action == 'w':
                    np.save(filename, self.__dict__[key])
                else:
                    self.__dict__[key] = np.load(filename)

    def __init__(self, calibration=None, input_folder=None):
        """
        Initialize camera calibration.

        If another calibration object is provided, copy its values. If an input
        folder is provided, load ``*.npy`` files from that folder. An input
        folder overwrites a calibration object.
        """
        #: Camera matrices (M)
        self.cam_mats = {"left": None, "right": None}
        #: Distortion coefficients (D)
        self.dist_coefs = {"left": None, "right": None}
        #: Rotation matrix (R)
        self.rot_mat = None
        #: Translation vector (T)
        self.trans_vec = None
        #: Essential matrix (E)
        self.e_mat = None
        #: Fundamental matrix (F)
        self.f_mat = None
        #: Rectification transforms (3x3 rectification matrix R1 / R2)
        self.rect_trans = {"left": None, "right": None}
        #: Projection matrices (3x4 projection matrix P1 / P2)
        self.proj_mats = {"left": None, "right": None}
        #: Disparity to depth mapping matrix (4x4 matrix, Q)
        self.disp_to_depth_mat = None
        #: Bounding boxes of valid pixels
        self.valid_boxes = {"left": None, "right": None}
        #: Undistortion maps for remapping
        self.undistortion_map = {"left": None, "right": None}
        #: Rectification maps for remapping
        self.rectification_map = {"left": None, "right": None}
        #: Homography matrix for projection from one camera to other
        self.homography_mat = {"left": None, "right": None}
        #: Homography matrix for projection from one undistorted camera image to another
        self.undistorted_homography_mat = {"left": None, "right": None}
        #: Image sizes for left and right images
        self.calibrationImgShape = None

        if calibration:
            self._copy_calibration(calibration)
        elif input_folder:
            self.load(input_folder)

    def load(self, input_folder):
        """Load values from ``*.npy`` files in ``input_folder``."""
        self._interact_with_folder(input_folder, 'r')

    def export(self, output_folder):
        """Export matrices as ``*.npy`` files to an output folder."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self._interact_with_folder(output_folder, 'w')

    def rectify(self, frames):
        """
        Rectify frames passed as (left, right) pair of OpenCV Mats.

        Remapping is done with nearest neighbor for speed.
        """
        new_frames = []
        for i, side in enumerate(("left", "right")):
            new_frames.append(cv2.remap(frames[i],
                                        self.undistortion_map[side],
                                        self.rectification_map[side],
                                        cv2.INTER_NEAREST))
        return new_frames


class StereoCalibrator(object):

    """A class that calibrates stereo cameras by finding chessboard corners."""

    def _get_corners(self, image):
        """Find subpixel chessboard corners in image."""
        
        #convert image to black and white
        #temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        temp = image

        #if necessary, resize the image for display on screen and to reduce initial guess time
        temp_resized, scale = self._resize_image(temp)

        #use quick check of image corners to get initial guess
        ret, corners = cv2.findChessboardCorners(temp_resized,
                                                 (self.rows, self.columns), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK)
        if not ret:
            #try without fast check flag
            ret, corners = cv2.findChessboardCorners(temp_resized,
                                                 (self.rows, self.columns), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ret:
                #no corners found
                return None

        #rescale corners using applied resizing scale
        corners = corners*scale #[c*scale for c in corners]

        #temporarily suppres to check improvement in speed
        cv2.cornerSubPix(temp, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                          30, 0.01))
        return corners

    def _resize_image(self, image, scaling = 1, target_pix_dim = 800.0):
        """
        Resize image to a smaller size based on max_pix_dim
        (max number of pixels in any dimension)
        """
        temp = image
        #get x and y dimensions of the image
        px, py = temp.shape[:2]

        #check if the image is already close enough to target dim
        if abs(max(px,py) - target_pix_dim) < 100:
            #if the image doesn't need any scaling, return it unmodified
            return [temp, scaling]
        else:
            scaling = max(px, py)/800.0
        
        #get new scaled image dimensions
        #it's ok if it gets slightly warped since this is only used
        #for display purposes and finding the initial chessboard guess
        fy,fx = [int(s/scaling) for s in temp.shape[:2]]

        #resize the image
        temp = cv2.resize(temp, (fx, fy), interpolation = cv2.INTER_AREA)

        return [temp, scaling]

    def _show_corners(self, image, corners):
        """Show chessboard corners found in image."""
        temp = image
        cv2.drawChessboardCorners(temp, (self.rows, self.columns), corners,
                                  True)
        window_name = "Chessboard"

        temp, scaling = self._resize_image(temp)         

        cv2.imshow(window_name, temp)
        if cv2.waitKey(0):
            cv2.destroyWindow(window_name)

    def __init__(self, rows, columns, square_size):
        """
        Store variables relevant to the camera calibration.

        ``corner_coordinates`` are generated by creating an array of 3D
        coordinates that correspond to the actual positions of the chessboard
        corners observed on a 2D plane in 3D space.
        """
        #: Number of calibration images
        self.image_count = 0
        #: Number of inside corners in the chessboard's rows
        self.rows = rows
        #: Number of inside corners in the chessboard's columns
        self.columns = columns
        #: Size of chessboard squares in cm
        self.square_size = square_size
        #: Size of calibration images in pixels
        self.image_size = None

        #size of left and right images used to stereocalibrate
        self.image_shapes = {'left': None, 'right': None}

        pattern_size = (self.rows, self.columns)
        corner_coordinates = np.zeros((np.prod(pattern_size), 3), np.float32)
        corner_coordinates[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        corner_coordinates *= self.square_size
        #: Real world corner coordinates found in each image
        self.corner_coordinates = corner_coordinates
        #: Array of real world corner coordinates to match the corners found
        self.object_points = []
        #: Array of found corner coordinates from calibration images for left
        #: and right camera, respectively
        self.image_points = {"left": [], "right": []}
        self.undistorted_image_points = {'left': [], 'right': []}
        #: List to record image pairs that weren't able to be used in calibration
        self.bad_images = []
        #: List to record image pairs that were able to be used in calibration
        self.good_images = []

    def add_corners(self, image_pair, show_results=False, undistorted = False):
        """
        Record chessboard corners found in an image pair.

        The image pair should be an iterable composed of two CvMats ordered
        (left, right).
        """
        if self.image_size == None:
            (height, width) = image_pair[0].shape
            self.image_size = (width, height)

        side = "left"
        good_pair = True #track if corners found in both images
        image_points = {'left':None, 'right':None} #store until both images confirmed good
        for image in image_pair:
            corners = self._get_corners(image)

            if corners is None:
                #no corners were found
                #return error info in form of 
                #append failed side to return status message
                good_pair = False
                break
            else:
                #if corners found in this image and no error in prior image in pair
                image_points[side] = corners.reshape(-1,2)

                if show_results:
                    self._show_corners(image, corners)

            side = "right"

        if good_pair:
            #if we found corners in both images,
            #append to list of image points for calibration
            for s in ['left', 'right']:
                if undistorted:
                    self.undistorted_image_points[s].append(image_points[s])
                else:
                    self.image_points[s].append(image_points[s])

            self.object_points.append(self.corner_coordinates)
            self.image_count += 1

        return good_pair

    def _returnJPGorientation(self, file):
        #return exif orientation information
        orientation_key = 274
        img = Image.open(file)

        try:
            orientation = img._getexif()[orientation_key]
        except:
            orientation = None

        return orientation

    def _rotate90(self, img, n):
        ##rotate image 90 degrees CCW n times

        assert(type(n) is int)
        assert(n > 0)

        n = n%4

        #create indices to transpose array
        #i.e. don't make nxmx3 RGB array into 3xmxn array 
        indices = [i for i in range(len(img.shape))]
        indices[0], indices[1] = indices[1], indices[0]
        indices = tuple(indices)

        if n == 1:
            #rotate 90 deg CCW by transposing, then flipping
            img = np.flipud(np.transpose(img, axes = indices))
        elif n == 2:
            #rotate 180 degrees CCW
            img = np.flipud(np.fliplr(img))
        elif n == 3:
            #rotate 270 degrees CCW
            #by flipping vertically, then transposing
            img = np.transpose(np.flipud(img), axes = indices)

        return img

    def read_JPG_ignoring_orientation(self, file):
        #read image using cv2.imread()
        #reverse any orientation compensation to return
        #image with unmodified orientation

        if orientation is None or orientation == 1:
            return img
        if orientation == 2:
            #horizontal flip
            img = np.fliplr(img)
        if orientation == 3:
            #rotate 180 deg
            img = self._rotate90(img, 2)
        elif orientation == 4:
            #flip vertically
            img = np.flipud(img)
        elif orientation == 5:
            #flip horizontally
            #then rotate 90 deg CCW
            img = np.flipud(img)
            img = self._rotate90(img, 1)
        elif orientation == 6:
            #rotate 90 deg CCW
            img = self._rotate90(img, 1)
        elif orientation == 7:
            #flip horizontal
            #then rotate 90 deg CW
            img = np.fliplr(img)
            img = self._rotate90(img, 3)
        elif orientation == 8:
            #rotate 90 degrees CW
            img = self._rotate90(img, 3)

        return img

    def calibrate_cameras(self):
        """Calibrate cameras based on found chessboard corners."""
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                    100, 1e-5)
        flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +
                 cv2.CALIB_SAME_FOCAL_LENGTH)
        calib = StereoCalibration()
        (calib.cam_mats["left"], calib.dist_coefs["left"],
         calib.cam_mats["right"], calib.dist_coefs["right"],
         calib.rot_mat, calib.trans_vec, calib.e_mat,
         calib.f_mat) = cv2.stereoCalibrate(self.object_points,
                                            self.image_points["left"],
                                            self.image_points["right"],
                                            self.image_size,
                                            calib.cam_mats["left"],
                                            calib.dist_coefs["left"],
                                            calib.cam_mats["right"],
                                            calib.dist_coefs["right"],
                                            calib.rot_mat,
                                            calib.trans_vec,
                                            calib.e_mat,
                                            calib.f_mat,
                                            criteria=criteria,
                                            flags=flags)[1:]
        (calib.rect_trans["left"], calib.rect_trans["right"],
         calib.proj_mats["left"], calib.proj_mats["right"],
         calib.disp_to_depth_mat, calib.valid_boxes["left"],
         calib.valid_boxes["right"]) = cv2.stereoRectify(calib.cam_mats["left"],
                                                      calib.dist_coefs["left"],
                                                      calib.cam_mats["right"],
                                                      calib.dist_coefs["right"],
                                                      self.image_size,
                                                      calib.rot_mat,
                                                      calib.trans_vec,
                                                      flags=0)
        for side in ("left", "right"):
            (calib.undistortion_map[side],
             calib.rectification_map[side]) = cv2.initUndistortRectifyMap(
                                                        calib.cam_mats[side],
                                                        calib.dist_coefs[side],
                                                        calib.rect_trans[side],
                                                        calib.proj_mats[side],
                                                        self.image_size,
                                                        cv2.CV_32FC1)
        # This is replaced because my results were always bad. Estimates are
        # taken from the OpenCV samples.
        width, height = self.image_size
        focal_length = 0.8 * width
        calib.disp_to_depth_mat = np.float32([[1, 0, 0, -0.5 * width],
                                              [0, -1, 0, 0.5 * height],
                                              [0, 0, 0, -focal_length],
                                              [0, 0, 1, 0]])

        #pass size of images used to calibrate to the calib object
        calib.calibrationImgShape = self.image_shapes

        #calculate the Homography matrix from src image to dest image
        #perform and store for both left to right and right to left
        for side in (('left', 'right'),('right', 'left')):
            src = side[0]
            dest = side[1]
            calib.homography_mat[src] = self.returnHomographyMatrix(self.image_points, src_key = src, dest_key = dest)

        return calib

    def returnHomographyMatrix(self, image_points, src_key = 'right', dest_key = 'left'):
        '''
        Return homography matrix warping pixels from src image to dest image.
        Use later with cv2.warpPerspective to overlay images.
        Default args assume left image is the destination.
        Pixels outside of radius of ransacReprojThreshold (in pixels)
        are considered outliers and ignored during computation.
        '''
        #concatenate all lists of points
        src_points = np.vstack(image_points[src_key])
        dest_points = np.vstack(image_points[dest_key])

        h, mask = cv2.findHomography(src_points, dest_points)#,
            #cv2.CV_RANSAC, 8)

        if len(h) == 0:
            raise RuntimeError('Could not calculate homography matrix for src = {0} and dest = {1}'.format(src_key, dest_key))

        return h

    def check_calibration(self, calibration):
        """
        Check calibration quality by computing average reprojection error.

        First, undistort detected points and compute epilines for each side.
        Then compute the error between the computed epipolar lines and the
        position of the points detected on the other side for each point and
        return the average error.
        """
        sides = "left", "right"
        which_image = {sides[0]: 1, sides[1]: 2}
        undistorted, lines = {}, {}
        for side in sides:
            undistorted[side] = cv2.undistortPoints(
                         np.concatenate(self.image_points[side]).reshape(-1,
                                                                         1, 2),
                         calibration.cam_mats[side],
                         calibration.dist_coefs[side],
                         P=calibration.cam_mats[side])
            lines[side] = cv2.computeCorrespondEpilines(undistorted[side],
                                              which_image[side],
                                              calibration.f_mat)
        total_error = 0
        this_side, other_side = sides
        for side in sides:
            for i in range(len(undistorted[side])):
                total_error += abs(undistorted[this_side][i][0][0] *
                                   lines[other_side][i][0][0] +
                                   undistorted[this_side][i][0][1] *
                                   lines[other_side][i][0][1] +
                                   lines[other_side][i][0][2])
            other_side, this_side = sides
        total_points = self.image_count * len(self.object_points)
        return total_error / total_points

    def overlay_corners(self, calibration):
        #reproject all points using homography matrix
        left_shape = self.image_shapes['left']
        right_shape = self.image_shapes['right']
        left = np.ones(left_shape, np.uint8)*255
        right = np.zeros(right_shape, np.uint8)*255
        left = np.repeat(left[:,:,np.newaxis], 3, axis=2)
        right = np.repeat(right[:,:,np.newaxis], 3, axis=2)
        left_points = np.concatenate(self.image_points['left'])
        right_points = np.concatenate(self.image_points['right'])

        for p in left_points:
            cv2.circle(left, tuple(p), 50, (255,0,0), thickness = 3)
        for p in right_points:
            cv2.circle(right, tuple(p), 50, (0,255,0), thickness = 3)

        #project right points to left image frame
        warped_right = cv2.warpPerspective(right, calibration.homography_mat['right'], (left.shape[1], left.shape[0]))

        #combine left and right image points into single image
        combined = cv2.addWeighted(left, 1, warped_right, 1, 0)

        #resize to fit on the screen
        scale = 1000./combined.shape[1]
        dims = (int(combined.shape[1]*scale), int(combined.shape[0]*scale))
        stacked = cv2.resize(combined, dims)

        cv2.imshow('All Image Points In Left Image Frame', stacked)
        if cv2.waitKey(0):
            cv2.destroyWindow('All Points')

    def overlay_calibration_images(self,calibration):

        progress = ProgressBar(maxval=len(self.good_images),
                          widgets=[Bar("=", "[", "]"),
                          " ", Percentage()])
        print('Overlaying right images onto left images.')
        progress.start()

        i=0
        while i < len(self.good_images):
            left, right = self.good_images[i:i+2]

            #img_left, img_right = cv2.imread(left, cv2.CV_LOAD_IMAGE_GRAYSCALE), cv2.imread(right, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img_left, img_right = self.read_JPG_ignoring_orientation(left), self.read_JPG_ignoring_orientation(right)

            #convert raw array to float64 type due to odd bug in warp perspective
            projected_right = cv2.warpPerspective(img_right.astype(np.float_),
                calibration.homography_mat['right'],
                (img_left.shape[1],img_left.shape[0]))
            projected_right = projected_right.astype(np.uint8)

            #overlay left and right image
            overlaid_image = cv2.addWeighted(img_left, 0.5,
                projected_right, 0.5, 0)

            #resize for the screen
            scale = 1000./overlaid_image.shape[1]
            dims = (int(overlaid_image.shape[1]*scale), int(overlaid_image.shape[0]*scale))
            overlaid_image = cv2.resize(overlaid_image, dims)

            cv2.imshow('Overlay: {0}'.format(left), overlaid_image)
            if cv2.waitKey(0):
                cv2.destroyWindow('Overlay: {0}'.format(left))

            i += 2

            progress.update(i)

        progress.finish()
