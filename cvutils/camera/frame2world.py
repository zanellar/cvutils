# -*- encoding: utf-8 -*-
import numpy as np

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇


def frame2plane(frame):
    normal = frame.M.UnitZ()
    a = normal.x()
    b = normal.y()
    c = normal.z()
    d = -(a * frame.p.x() + b * frame.p.y() + c * frame.p.z())
    return np.array([a, b, c, d]).reshape(4)


def image2cartesian(image_point, camera, plane_coefficients=[0, 0, 0, 0]):
    # from image point [pixels] to cartasian ray [meters]
    image_point = np.array([
        image_point[0],
        image_point[1],
        1.0
    ]).reshape(3, 1)
    ray = np.matmul(camera.camera_matrix_inv, image_point)
    ray = ray / np.linalg.norm(ray)
    ray = ray.reshape(3)

    # intersection cartesian ray and target plane
    # plane_coefficients = frame2plane(frame)
    t = -(plane_coefficients[3]) / (
        plane_coefficients[0] * ray[0] +
        plane_coefficients[1] * ray[1] +
        plane_coefficients[2] * ray[2]
    )
    x = ray[0] * t
    y = ray[1] * t
    z = ray[2] * t
    plane_point = np.array([x, y, z])
    plane_point = plane_point.reshape(3)

    return plane_point


def camera2fixed(point, camera):
    return np.matmul(camera.R, point) - camera.T
