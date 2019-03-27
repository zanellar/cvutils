import cv2
import numpy as np
import transforms3d as tf


def draw2DReferenceFrame(frame, image, size=-1, thick=2, tipLength=0.3, force_color=None):
    if size < 0:
        size = image.shape[0] * 0.1
    p, M, _, _ = tf.affines.decompose(frame)

    center = np.array([p[0], p[1]], dtype=int)
    ax = np.array([M[0, 0], M[1, 0]], dtype=float) * size  # BUG
    ay = np.array([M[0, 1], M[1, 1]], dtype=float) * size
    ax = ax.astype(int)
    ay = ay.astype(int)
    c1 = center + ax
    c2 = center + ay

    color_x = (0, 0, 255)
    color_y = (0, 255, 0)
    if force_color != None:
        color_x = color_y = force_color
    try:
        cv2.arrowedLine(image, tuple(center), tuple(c1),
                        color_x, thick, tipLength=tipLength)
        cv2.arrowedLine(image, tuple(center), tuple(c2),
                        color_y, thick, tipLength=tipLength)
    except Exception as e:
        print(e)
