#!/usr/bin/env python
# -*- encoding: utf-8 -*-


''' Example of marker detection with ArUco from image file. 
 * Define the area of interest w.r.t. the marker.
 * Draw the marker bourders, its 2D frame and the respective area of interest on the image.
 * Do some image processing on the area of interest
 '''

import math
import numpy as np
import argparse
import transforms3d as tf
from transforms3d.derivations import eulerangles as tfeuler

import cv2
from cvutils.markers.detector import MarkerDetector
from cvutils.markers.area import ImageAreaOfInterest
from cvutils.markers.markersutils import draw2DReferenceFrame

from cvutils.processing.processingutils import ImageProcessing


# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


ap = argparse.ArgumentParser()
ap.add_argument("--image_file", required=True, help="Target image file.")
ap.add_argument("--marker_id", default=[777, 666])
ap.add_argument("--marker_size", default=[0.01, 0.01])
ap.add_argument("--debug", default=False, type=bool)
args = ap.parse_args()

camera_matrix = np.matrix([[630.826117, 0, 334.029946], [0, 630.425892,  230.675118], [0, 0, 1.0]])
camera_distorsion = np.array([0.029701, 0.277058, 0.00059, 0.003557, -1.293219])
camera_params = dict(camera_matrix=camera_matrix, camera_distorsion=camera_distorsion)

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


class MarkerHandler(object):
    def __init__(self, camera_params):
        self.camera_matrix = camera_params["camera_matrix"]
        self.camera_distorsion = camera_params["camera_distorsion"]
        self.image = None
        self.pixel_ratio = 0
        self.marker_tf_2d = None
        self.markers = None

        # Creates marker detector
        self.marker_detector = MarkerDetector(self.camera_matrix, self.camera_distorsion, z_up=True)

    def detect(self, image, markers_map, camera_tf_name=None, only_first=False):
        ''' Detects the markers on the image.
        \nIt recieves a map with the marker 'id' as 'key' and the marker 'size' as 'value'
        \nIt returns a map with the marker 'id' as 'key' and the marker 'tf' as 'value'. '''
        self.image = image
        if image is None:
            return

        self.markers = self.marker_detector.detectMarkersMap(self.image, markers_map=markers_map)

        if only_first and bool(self.markers):
            for id, marker in self.markers.items():
                print("MARKER", marker)
                self.markers[id] = marker
                break

        self.marker_tf_2d = dict.fromkeys(self.markers)
        self.pixel_ratio = dict.fromkeys(self.markers)
        self.marker_centers = dict.fromkeys(self.markers)

        if bool(self.markers):
            for id, marker in self.markers.items():
                self.pixel_ratio[id] = marker.measurePixelRatio(markers_map[id])  # compute meters/pixels ratio
                self.marker_tf_2d[id] = marker.get2DFrame()  # 2D tf of the marker (tf on the marker's plane)
                self.marker_centers[id] = marker.center

            return self.markers

    def isDetected(self):
        return bool(self.markers)

    def draw(self):
        ''' Draw the marker bourders and its reference frame on the source image'''
        if bool(self.markers):
            for id, marker in self.markers.items():
                marker.draw(image=self.image, color=(255, 0, 0), scale=1, draw_center=True)
                draw2DReferenceFrame(self.marker_tf_2d[id], self.image)

    def center(self):
        if bool(self.markers):
            return self.marker_centers


# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# idyframe = tf.affines.compose(np.zeros(3),np.identity(3),np.ones(3))

if __name__ == '__main__':

    markers_map = {args.marker_id[0]: args.marker_size[0],
                   args.marker_id[1]: args.marker_size[1]
                   }

    # ⬢⬢⬢⬢⬢➤ MARKER
    marker = MarkerHandler(camera_params)

    # ▇▇▇▇▇▇▇▇▇ LOOP ▇▇▇▇▇▇▇▇▇

    while True:
        image = cv2.imread(args.image_file)

        if image is not None:
            original_image_copy = image.copy()

            # detect the markers on the image
            detections = marker.detect(image=image,
                                       markers_map=markers_map)

            if marker.isDetected():

                print(marker.center())

                # ⬢⬢⬢⬢⬢➤ Image Area of Interest
                for id in detections.keys():
                    image_area = ImageAreaOfInterest()  # BUG TRANSFORMATION & COORDINATES
                    relative_tf_2d = tf.affines.compose([50, 50, 0], tfeuler.z_rotation(-np.pi / 2.0), np.ones(3))
                    image_area.update(image, marker.marker_tf_2d[id], relative_tf_2d)
                    image_area.setRectArea(height=300, width=300)

                    masked_image = image_area.getMaskedImage(show_it=args.debug)
                    # cropped_image = image_area.getCroppedArea(show_it=args.debug) # BUG size.width<0 or size.height<0

                    # ⬢⬢⬢⬢⬢➤ Image Processing
                    # image_proc = cropped_image
                    # image_proc = ImageProcessing.grayScale(image_proc, filter_it=True, show_it=args.debug)
                    # image_proc = ImageProcessing.adaptiveThreshold(image_proc, show_it=args.debug)
                    # image_proc = ImageProcessing.saltAndPepper(image_proc, kerner_size=3, show_it=args.debug)
                    # image_proc = ImageProcessing.fixedThreshold(image_proc, th=200, show_it=args.debug)
                    # image_proc = ImageProcessing.templateMatch(image_proc, show_it=args.debug)

                    # ⬢⬢⬢⬢⬢➤ Augmented reality on image
                    image_area.draw()
                    marker.draw()

            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow("output", image)

            c = cv2.waitKey(1)
