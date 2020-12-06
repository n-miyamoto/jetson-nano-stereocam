import cv2
import numpy as np
import glob
import pylab as plt

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


crit = 5

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

IMAGE_WIDTH = 3280
IMAGE_HEIGHT= 2464

# path of calibration images
PATH = "./calib_image_stereo/"

def show_result(im_l,im_r,disp):
    graph = plt.figure()
    plt.rcParams["font.size"]=5
    # left image
    plt.subplot(2,2,1),plt.imshow((im_l))
    plt.title("Left Image")
    # right image
    plt.subplot(2,2,2),plt.imshow((im_r))
    plt.title("Right Image")
    # disparity
    plt.subplot(2,2,3),plt.imshow(disp,"gray")
    plt.title("Disparity")
    plt.show()


def main():
    cap_left  = cv2.VideoCapture(GST_STR_L, cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(GST_STR_R, cv2.CAP_GSTREAMER)

    # wait 60 frames
    for i in range(20):
        ret, img_l = cap_left.read()
        ret, img_r = cap_right.read()

    # read left frame
    ret, image_left= cap_left.read()
    if ret != True:
        return 

    # read right frame
    ret, image_right= cap_right.read()
    if ret != True:
        return 

    # load camera parameters
    R = np.load(PATH + "R.npy")
    T = np.load(PATH + "T.npy")
    A1 = np.load(PATH + "A1.npy")
    A2 = np.load(PATH + "A2.npy")
    D1 = np.load(PATH + "D1.npy")
    D2 = np.load(PATH + "D2.npy")

    print("A1 = \n", A1)
    print("A2 = \n", A2)
    print("D1 = \n", D1)
    print("D2 = \n", D2)
    print("R = \n", R)
    print("T = \n", T)

    # rectify images
    flags = 0
    alpha = 0
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        A1, D1, A2, D2, image_size, R, T, flags, alpha, image_size)

    # undistortion and get rectify map
    m1type = cv2.CV_32FC1
    map1_l, map2_l = cv2.initUndistortRectifyMap(A1, D1, R1, P1, image_size, m1type)
    map1_r, map2_r = cv2.initUndistortRectifyMap(A2, D2, R2, P2, image_size, m1type)

    # load images
    gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    resized_left = cv2.resize(gray_left,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
    resized_right = cv2.resize(gray_right,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
    cv2.imshow('Left Target Image', resized_left)
    cv2.waitKey(0)
    cv2.imshow('Right Target Image', resized_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # ReMap images
    interpolation = cv2.INTER_NEAREST
    rectified_image_left  = cv2.remap(gray_left,  map1_l, map2_l, interpolation)
    rectified_image_right = cv2.remap(gray_right, map1_r, map2_r, interpolation)
    resized_left = cv2.resize(rectified_image_left,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
    resized_right = cv2.resize(rectified_image_right,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))

    cv2.imshow('Rectified Left Target Image', resized_left)
    cv2.waitKey(0)
    cv2.imshow('Rectified Right Target Image', resized_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # stereo block matching
    # matcher = cv2.StereoBM_create(numDisparities = 256,blockSize = 15)
    # disparity = matcher.compute(rectified_image_left, rectified_image_right)
    # map = ( disparity - np.min(disparity) ) / ( np.max(disparity) - np.min(disparity) )
    # resized_map = cv2.resize(map,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
    # cv2.imshow('disparity', resized_map)
    # cv2.waitKey(0)

    # stereo semi global block matching
    matcher = cv2.StereoSGBM_create(minDisparity=10, numDisparities=256, blockSize=22)
    disparity = matcher.compute(rectified_image_left, rectified_image_right)
    print(disparity)
    print(np.max(disparity))
    print(np.min(disparity))
    map = ( disparity - np.min(disparity) ) / ( np.max(disparity) - np.min(disparity) )
    print(map)
    print(np.max(map))
    print(np.min(map))
    # resized_map = cv2.resize(map,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
    # cv2.imshow('disparity', resized_map)
    # cv2.waitKey(0)

    # show result
    min_disp = 10
    show_result(rectified_image_left,rectified_image_right,(disparity-min_disp)/(256-min_disp))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
