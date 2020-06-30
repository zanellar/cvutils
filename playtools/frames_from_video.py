
import os
import cv2
print(cv2.__version__)

video_file_path = '/home/riccardo/Downloads/VID_20200630_162250.mp4'
save_images_folder_path = '/media/riccardo/data1/datasets/dlo_segmentation/data_chromakey/ck_raw/orange'

vidcap = cv2.VideoCapture(video_file_path)
sample_period = 10
success, image = vidcap.read()
count, ind = 0, 0
success = True
while success:
    if count % sample_period == 0:
        ind += 1
        image_file_path = os.path.join(save_images_folder_path, "frame%d.png" % ind)
        cv2.imwrite(image_file_path, image)
        print('Read a new frame: ', success, count, ind)
    success, image = vidcap.read()
    count += 1
