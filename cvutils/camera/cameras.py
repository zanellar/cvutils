# -*- encoding: utf-8 -*-
"""Cameras management."""
import cv2
import numpy as np

import os
import sys
import json


# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇

class Camera(object):

    def __init__(self, configuration_json):
        self.configuration_file = configuration_json

        if not os.path.exists(self.configuration_file):
            print("Camera configuration file not found! '{}'".format(self.configuration_file))
            sys.exit(0)

        camera_calibration = json.loads(open(self.configuration_file).read())

        self.fixed_frame = np.array(list(camera_calibration["fixed_frame_name"]))
        self.position_vector = np.array(list(camera_calibration["position_vector"]))
        self.rotation_matrix = np.array(list(camera_calibration["rotation_matrix"]))
        self.camera_matrix = np.array(list(camera_calibration["camera_matrix"]))
        self.distortion_coefficients = np.array(list(camera_calibration["dist_coeffs"]))
        self.image_size = np.array(list(camera_calibration["image_size"]))

        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.width = int(self.image_size[0])
        self.height = int(self.image_size[1])
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[1][2]
        self.k1 = self.distortion_coefficients[0]
        self.k2 = self.distortion_coefficients[1]
        self.p1 = self.distortion_coefficients[2]
        self.p2 = self.distortion_coefficients[3]
        self.k3 = self.distortion_coefficients[4]

    def getCameraFile(self):
        return self.configuration_file

    def get3DPointRGBD(self, u, v, framergbd):
        d = float(framergbd.depth_image[u, v]) / float(framergbd.depth_scale)
        px = (d / self.fx) * (v - self.cx)
        py = (d / self.fy) * (u - self.cy)
        pz = d
        point = np.array([px, py, pz])
        return point

    def get3DPointPlane(self, u, v, plane_coefficients, fixed_frame=False, reference_frame=None, plane_wrt_world=False):
        '''plane_coefficients = [a,b,c,d]
        plane equation: ax+by+cz+d=0'''

        camera_frame = np.concatenate((self.rotation_matrix, self.position_vector.reshape(3, 1)), axis=1)
        camera_frame_h = np.concatenate((camera_frame, [[0, 0, 0, 1]]))

        # from image point [pixels] to cartasian ray [meters]
        image_point = np.array([
            u,
            v,
            1.0
        ]).reshape(3, 1)
        ray = np.matmul(self.camera_matrix_inv, image_point)
        ray = ray / np.linalg.norm(ray)
        ray = ray.reshape(3)

        # intersection cartesian ray and target plane
        if plane_wrt_world:  # TODO this must be tested
            plane_coefficients = np.matmul(np.matrix.transpose(camera_frame_h), plane_coefficients)

        t = -(plane_coefficients[3]) / (
            plane_coefficients[0] * ray[0] +
            plane_coefficients[1] * ray[1] +
            plane_coefficients[2] * ray[2]
        )
        x = ray[0] * t
        y = ray[1] * t
        z = ray[2] * t
        plane_point = np.array([x, y, z])

        # point with respect to the fixed reference frame
        if fixed_frame:
            # plane_point = np.matmul(np.linalg.inv(self.rotation_matrix), plane_point - self.position_vector)
            # plane_point = np.matmul(self.rotation_matrix, plane_point) + self.position_vector
            plane_point_h = np.concatenate((plane_point, [1]))
            plane_point_h = np.matmul(camera_frame_h, plane_point_h)
            plane_point = plane_point_h[:3]

        # point with respect to a user-defined frame
        if reference_frame is not None:
            plane_point_h = np.concatenate((plane_point, [1]))
            plane_point_h = np.matmul(reference_frame, plane_point_h)
            plane_point = plane_point_h[:3]

        return plane_point.reshape(3)

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


class FrameRGBD(object):

    def __init__(self, rgb_image=None, depth_image=None, time=None):
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.depth_scale = 1
        self.time = time
        if self.rgb_image == None or self.depth_image == None:
            self.valid = False
        else:
            self.valid = True

    def isValid(self):
        return self.valid

    def getPointCloud(self, camera, mask=None):
        points = []
        if self.isValid():
            for u in range(0, self.rgb_image.shape[0]):
                for v in range(0, self.rgb_image.shape[1]):
                    p = camera.get3DPoint(u, v, self)
                    points.append(p)
        return points


# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


class FrameRGB(object):

    def __init__(self, rgb_image=None, time=None):
        self.rgb_image = rgb_image
        self.time = time
        if self.rgb_image == None:
            self.valid = False
        else:
            self.valid = True

    def isValid(self):
        return self.valid
