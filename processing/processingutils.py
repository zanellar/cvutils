# -*- encoding: utf-8 -*-
""" Image Processing """

from __future__ import division, print_function
import math
import transforms3d as tf
import cv2
import numpy as np

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# idyframe = tf.affines.compose(np.zeros(3),np.identity(3),np.ones(3))


class ImageProcessing(object):

    @staticmethod
    def extractPoints(image, th=200):
        points = []
        max_x = 0
        max_x_point = None
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j] > th:
                    if j > max_x:
                        max_x = j
                        max_x_point = np.array([j, i])
                    points.append(np.array([j, i]))

        if len(points) <= 0:
            return [], np.array([0, 0])
        else:
            return points, max_x_point

    @staticmethod
    def grayScale(image, filter_it=False, show_it=False):
        '''Gray scale image'''
        # grasy scale conversion
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # filter
        if filter_it:
            kernel = np.ones((5, 5), np.float32) / 25
            image_gray = cv2.filter2D(image_gray, -1, kernel)

        if show_it:
            cv2.namedWindow("image_gray", cv2.WINDOW_NORMAL)
            cv2.imshow("image_gray", image_gray)

        return image_gray

    @staticmethod
    def fixedThreshold(image, th, show_it=False):
        '''Apply a fixed threshold'''
        image[image < th] = [0]
        if show_it:
            cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
            cv2.imshow("threshold", image)
        return image

    @staticmethod
    def saltAndPepper(image, th=200, kerner_size=2, show_it=False):
        '''Salt and pepper filter'''
        kernel = np.ones((kerner_size, kerner_size), np.uint8)
        cropped = cv2.erode(image, kernel)
        image_c = cv2.dilate(cropped, kernel)
        if show_it:
            cv2.namedWindow("salt_and_pepper", cv2.WINDOW_NORMAL)
            cv2.imshow("salt_and_pepper", image_c)
        return image_c

    @staticmethod
    def adaptiveThreshold(image, show_it=False):
        '''Apply an adaptive gaussian threshold'''
        # Adaptive Threshold on the image for backgorund removal
        th = cv2.adaptiveThreshold(image,
                                   255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   11,
                                   2)
        th[th == 255] = [50]
        th[th == 0] = [255]
        if show_it:
            cv2.namedWindow("adaptive_threshold", cv2.WINDOW_NORMAL)
            cv2.imshow("adaptive_threshold", th)
        return th

    @staticmethod
    def cropImage(image, x1, x2, y1, y2, show_it=False):
        '''Crop the image'''
        try:
            crop = image[y1:y2, x1:x2]
            if show_it:
                cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
                cv2.imshow("crop", crop)
            return crop
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def rotateImage(image, angle, show_it=False):
        '''Rotate the image'''
        rows = image.shape[0]
        cols = image.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle * 180 / math.pi, 1)
        dst = cv2.warpAffine(image, M, (cols, rows))
        if show_it:
            cv2.namedWindow("rotate", cv2.WINDOW_NORMAL)
            cv2.imshow("rotate", dst)
        return dst

    @staticmethod
    def resizeImage(image, d, forced=False):
        shape = image.shape[0:2]  # takes only the image dimension (not the numbers of channels)
        image_c = image
        if forced:
            x_max, y_max = d
            if shape[0] > x_max or shape[1] > y_max:
                if shape[0] > shape[1]:
                    x = x_max
                    y = y_max
                else:
                    x = y_max
                    y = x_max
                image_c = cv2.resize(image, (x, y), interpolation=cv2.INTER_CUBIC)
        else:
            ratio = max(shape) / max(d)
            if ratio > 1:
                dim_scaled = np.roll(np.array(shape) / ratio, 1)
                dim_scaled = tuple(dim_scaled.astype(int))
                image_c = cv2.resize(image, dim_scaled, interpolation=cv2.INTER_CUBIC)

        return image_c
