
import os
import cv2
import numpy as np
print(cv2.__version__)

video_file_path = '/media/riccardo/UBUNTU 18_0/00088.MTS'
save_images_folder_path = '/media/riccardo/data1/datasets/dlo_segmentation/data_chromakey/ck_raw/CK_OBJECTS/apple'
sample_period = 10
rotate = False
resize = None  # (640, 360)
ind = 0

#######################################################################


def get_frame(vidcap):
    success, image = vidcap.read()
    if success:
        if rotate:
            pass
            # TODO
            # h, w = image.shape[:2]
            # M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=90, scale=1.)
            # image = cv2.warpAffine(tmp, M, (w, h))
        if resize is not None:
            image = cv2.resize(image, tuple(resize), interpolation=cv2.INTER_AREA)
    return success, image


#######################################################################

vidcap = cv2.VideoCapture(video_file_path)
success, image = get_frame(vidcap)
count = 0
success = True

while success:

    if count % sample_period == 0:
        ind += 1
        image_file_path = os.path.join(save_images_folder_path, "frame%d.jpg" % ind)
        cv2.imwrite(image_file_path, image)
        print('Read a new frame: ', success, count, ind)

    success, image = get_frame(vidcap)
    count += 1
