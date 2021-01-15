
import os
import glob
import cv2
import numpy as np
print(cv2.__version__)

path = '/home/riccardo/tests/dlo_segmentation/TEST_IMAGES/pollock_bg'
path1 = '/home/riccardo/tests/dlo_segmentation/TEST_IMAGES/pollock'
image_file_list = glob.glob(os.path.join(path, "*.*"))
ind = 1

for image_file in image_file_list:

    image = cv2.imread(image_file)
    image_file_path = os.path.join(path1, "pollock%d.jpg" % ind)
    cv2.imwrite(image_file_path, image)

    ind += 1
