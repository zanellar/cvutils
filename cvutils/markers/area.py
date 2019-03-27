# -*- encoding: utf-8 -*-
""" Image Processing """

from __future__ import division, print_function
import math
import transforms3d as tf
import cv2
import numpy as np
from cvutils.markers.markersutils import draw2DReferenceFrame

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# idyframe = tf.affines.compose(np.zeros(3),np.identity(3),np.ones(3))

# BUG TRANSFORMATION & COORDINATES


class ImageAreaOfInterest(object):
    # all the tf are 2D-frames on the marker plane
    def __init__(self, image=None, image_tf=None, relative_tf=None):
        self.image = image
        self.image_tf = image_tf
        self.relative_tf = relative_tf
        self.image_area = []

    def update(self, image, image_tf, relative_tf):
        ''' @image as Numpy Arrays with shape=(h,w,3) 
            @image_tf as transform3d frame'''
        self.image = image
        self.image_tf = image_tf
        self.relative_tf = relative_tf
        self.image_area_tf = self.image_tf.dot(self.relative_tf)

    def setArea(self, height, width, base_side):
        ''' Define the area of interest '''
        bs = base_side
        pw = width
        ph = height / 2
        self.image_area = np.array([
            [0, 0],
            [bs, ph],
            [pw, ph],
            [pw, -ph],
            [bs, -ph]
        ])

    def setRectArea(self, height, width):
        self.setArea(height, width, base_side=0)

    def setPoligArea(self, image_area):
        ''' Define the area of interest '''
        self.image_area = image_area

    def draw(self):
        ''' Draw the area of interest and the relative reference frame on the source image'''
        draw2DReferenceFrame(self.image_area_tf, self.image)
        relative_image_area = self._getRelativeImageArea()
        drawPolygon(self.image, relative_image_area)

    def getMaskedImage(self, show_it=False):
        ''' Returns (and eventually shows) the image (Numpy array) with only the area of interest visible and black pixels outside it.'''
        mask = self._buildMask(self.image)
        masked = cv2.bitwise_and(self.image, self.image, mask=mask)  # TODO ???
        if show_it:
            cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
            cv2.imshow("masked", masked)

        return masked

    def getCroppedArea(self, show_it=False):  # TODO work in progress
        image = self.getMaskedImage()
        T, M, _, _ = tf.affines.decompose(self.image_tf)
        angle = tf.euler.mat2euler(M, axes='sxyz')[-1]
        image = ImageProcessing.rotateImage(image, angle=angle, show_it=False)
        area_coord = self._getRelativeImageArea()
        # M = np.array([[np.cos(angle), -np.sin(angle)],
        #               [np.sin(angle),  np.cos(angle)]])  # rotate 2D
        # area_coord = np.array(area_coord[1:])
        # area_coord[area_coord < 0] = 0

        x1, x2, y1, y2 = 0, 0, 0, 0
        for i in range(0, image.shape[0]):
            nonzero = np.nonzero(np.array(image[i, :]))[0]
            if np.size(nonzero) > 0:
                if x1 == 0 and x2 == 0 or x2 - x1 < self.image_area[2][0]:
                    x1 = np.amin(nonzero)
                    x2 = np.amax(nonzero)
                    y1 = i
            else:
                if x1 > 0 and x2 > 0:
                    y2 = i - 1
                    break
                continue

        cropped_image = ImageProcessing.cropImage(image, x1, x2, y1, y2,  show_it=show_it)

        return image

    def _buildMask(self, target_image):
        zeros = None
        if target_image is not None:
            zeros = np.zeros((target_image.shape[0], target_image.shape[1], 1), dtype=np.int8)
            points = self._getRelativeImageArea()
            drawPolygon(zeros, points, True, (255, 255, 255), -1)
        return zeros

    def _getRelativeImageArea(self, dtype=int):
        relative_image_area = []
        for p in self.image_area:
            pext = np.array([p[0], p[1], 0.0, 1.0])
            pext = self.image_area_tf.dot(pext)
            relative_image_area.append(np.array([
                pext[0],
                pext[1]
            ],  dtype=dtype))
        return relative_image_area

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


def drawPolygon(image, polygon_points, is_closed=True, color=(0, 0, 255), thickness=1):
    pts = []
    for p in polygon_points:
        pts.append((int(p[0]), int(p[1])))
    if thickness > 0:
        cv2.polylines(image, np.int32([pts]), isClosed=is_closed,
                      color=color, thickness=thickness)
    else:
        cv2.fillPoly(image, np.int32([pts]), color=color)
