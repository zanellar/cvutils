#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Markers detection."""

from __future__ import division, print_function
import math
import transforms3d as tf
from transforms3d.derivations import eulerangles as tfeuler
import cv2
from cv2 import aruco
import numpy as np

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


class ARMarker(object):

    def __init__(self, Rvec=None, Tvec=None, esqns=None, marker_id=None, z_up=False):
        super(ARMarker, self).__init__()
        self.Rvec = Rvec
        self.Tvec = Tvec[0]
        self.esqn = esqns
        self.esqns = esqns[0]
        self.marker_id = marker_id
        self.z_up = z_up
        rot, _ = cv2.Rodrigues(self.Rvec)

        self.M = np.array(                    # Is rotation of the frame, pertenece a Frame (self)
            [[rot[0, 0], rot[0, 1], rot[0, 2]],
             [rot[1, 0], rot[1, 1], rot[1, 2]],
             [rot[2, 0], rot[2, 1], rot[2, 2]]]
        )
        self.p = np.array(                      # Traslation of the frame
            [self.Tvec[[0]],
             self.Tvec[[1]],
             self.Tvec[[2]]]
        )
        if self.z_up:
            self.M = self.M*tfeuler.x_rotation(-np.pi / 2.0)
            self.M = self.M*tfeuler.z_rotation(-np.pi)

        self.corners = []
        for p in self.esqns:
            self.corners.append(p)

        self.radius = int(
            0.5 * np.linalg.norm(self.corners[0] - self.corners[2]))
        self.center = np.array([0.0, 0.0])
        for p in self.corners:
            self.center += p
        self.center = self.center / 4

        self.side_in_pixel = 0
        for i in range(0, len(self.corners)):
            i_next = (i + 1) % len(self.corners)
            p1 = np.array([self.corners[i]])
            p2 = np.array([self.corners[i_next]])
            self.side_in_pixel += np.linalg.norm(p1 - p2)
        self.side_in_pixel = self.side_in_pixel / float(len(self.corners))

    def getID(self):
        return self.marker_id[0]

    def getName(self):
        return "marker_{}".format(self.getID())

    def draw(self, image, color=(255, 0, 0), scale=1, draw_center=False):
        esqn_cnr = np.array([self.esqn])
        cv2.aruco.drawDetectedMarkers(image, esqn_cnr, self.marker_id, 1)
        if draw_center:
            cv2.circle(
                image,
                (int(self.center[0]), int(self.center[1])),
                radius=3,
                color=color,
                thickness=3 * scale
            )

    def get2DFrame(self):
        ax = np.array(self.corners[1] - self.corners[0])
        ax = ax / np.linalg.norm(ax)
        ay = np.array([ax[1], -ax[0]])
        return ARMarker._2tf3d(ax, ay, self.center)

    @staticmethod
    def _2tf3d(vx, vy, center):
        ''' transoform to tf3d a 2d rotation'''
        rot = np.array(
            [[vx[0], vy[0], 0.0],
             [vx[1], vy[1], 0.0],
             [0, 0, -1]]
        )
        frame = tf.affines.compose([center[0], center[1], 0], rot, [1, 1, 1])
        return frame

    def applyCorrection(self, frame):
        self_frame = tf.affines.compose(self.p, self.M, [1, 1, 1])
        self_frame = self_frame * frame
        self.M, self.p, _, _ = tf.affines.decompose(self_frame)

    def measurePixelRatio(self, side_in_meter):
        return side_in_meter / self.side_in_pixel

    # def getPlaneCoefficients(self,frame):
    #     return ARMarker._planeCoefficientsFromFrame(frame)

    # @staticmethod
    # def _planeCoefficientsFromFrame(frame):
    #     """ Builds 3D Plane coefficients centered in a Reference Frame """
    #     normal = frame.M.UnitZ() # TODO convert PyKDL to tf3d
    #     a = normal.x()
    #     b = normal.y()
    #     c = normal.z()
    #     d = -(a * frame.p.x() + b * frame.p.y() + c * frame.p.z())
    #     return np.array([a, b, c, d]).reshape(4)

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


class MarkerDetector(object):

    def __init__(self, camera_matrix, camera_distorsion, min_size=0.01, max_size=0.5, z_up=False):
        self.cameraMatrix = camera_matrix
        self.distCoeffs = camera_distorsion
        self.z_up = z_up
        self.index = 0

    def detectMarkers(self, image, markers_metric_size=-1.0, markers_map=None):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict)
        print(ids)
        final_markers = []
        if markers_metric_size < 0 and markers_map is None:
            return final_markers
        elif markers_map is not None:
            for marker_id in markers_map.keys():
                try:
                    marker_id = int(marker_id)
                except Exception as e:
                    print(e)

                if ids is not None:
                    detected_ids = ids.flatten()
                    if marker_id in detected_ids:
                        for i in detected_ids:
                            if i == marker_id:
                                esquinas = corners[self.index]
                                esqns = np.array([esquinas])
                                marker = ids[self.index]
                                break
                            self.index = self.index + 1
                        self.index = 0
                        estim_pos_marker = aruco.estimatePoseSingleMarkers(esqns, 0.01, self.cameraMatrix, self.distCoeffs)
                        rvecs, tvecs = estim_pos_marker[0:2]
                        # BUG cv-version='3.2.0-dev': estimatePoseSingleMarkers -> "rvecs, tvecs"
                        #     cv-version='3.3.1': estimatePoseSingleMarkers -> "rvecs, tvecs, _objPoints"

                        tvec = tvecs[0]
                        esqns = esqns[0]
                        final_markers.append(ARMarker(rvecs, tvec, esqns, marker, self.z_up))
        return final_markers

    def detectMarkersMap(self, image, markers_metric_size=-1.0, markers_map=None):
        markers = self.detectMarkers(image,
                                     markers_metric_size,
                                     markers_map)  # Returns an array of arrays (Rotation Vector M, and Traslation Vector p)
        markers_map = {}
        for marker in markers:
            markers_map[marker.getID()] = marker
        return markers_map  # markers_map is a dict of the ids detected
