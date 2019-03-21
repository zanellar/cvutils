import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('/home/riccardo/Downloads/cv_temp/2019-03-13-164630.webm')
success, image = vidcap.read()
count = 0
success = True
while success:
    if count >= 40:
        ind = count-40
        cv2.imwrite("frame%d.jpg" % ind, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
