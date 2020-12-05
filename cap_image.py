import cv2

GST_STR_L = 'nvarguscamerasrc sensor-id=0 \
        ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)20/1 \
        ! nvvidconv ! video/x-raw, width=(int)3280, height=(int)2464, format=(string)BGRx \
        ! videoconvert \
        ! appsink'

GST_STR_R = 'nvarguscamerasrc sensor-id=1 \
        ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)20/1 \
        ! nvvidconv ! video/x-raw, width=(int)3280, height=(int)2464, format=(string)BGRx \
        ! videoconvert \
        ! appsink'

WINDOW_NAME = 'Camera Test'

def main():
    cap_left  = cv2.VideoCapture(GST_STR_L, cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(GST_STR_R, cv2.CAP_GSTREAMER)

    # wait 60 frames
    for i in range(60):
        ret, img_l = cap_left.read()
        ret, img_r = cap_right.read()

    # read left frame
    ret, img_l = cap_left.read()
    if ret != True:
        return 

    # read right frame
    ret, img_r = cap_right.read()
    if ret != True:
        return 

    # show images
    cv2.imshow(WINDOW_NAME, img_l)
    key = cv2.waitKey()
    cv2.imshow(WINDOW_NAME, img_r)
    key = cv2.waitKey()

    # save images
    cv2.imwrite('./image_left.jpg', img_l)
    cv2.imwrite('./image_right.jpg', img_r)

if __name__ == "__main__":
    main()

