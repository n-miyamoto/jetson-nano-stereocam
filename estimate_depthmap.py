import cv2
import numpy as np
import glob

square_size = 2.2      # rect size [cm]
pattern_size = (10, 7)

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints_left = []
imgpoints_right = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

IMAGE_WIDTH = 3280
IMAGE_HEIGHT= 2464

# path of calibration images
PATH = "./calib_image_stereo/"

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

    # rectify images
    flags = 0
    alpha = 1
    image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        A1, D1, A2, D2, image_size, R, T, flags, alpha, image_size)

    # undistortion and get rectify map
    m1type = cv2.CV_32FC1
    map1_l, map2_l = cv2.initUndistortRectifyMap(A1, D1, R1, P1, image_size, m1type)
    map1_r, map2_r = cv2.initUndistortRectifyMap(A2, D2, R2, P2, image_size, m1type)

    # load images
    bgr_left = cv2.imread(image_left, cv2.IMREAD_COLOR)
    bgr_right = cv2.imread(image_right, cv2.IMREAD_COLOR)
    gray_left = cv2.cvtColor(bgr_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(bgr_right , cv2.COLOR_BGR2GRAY)

    # ReMap images
    interpolation = cv2.INTER_NEAREST
    rectified_image_left  = cv2.remap(gray_left,  map1_l, map2_l, interpolation)
    rectified_image_right = cv2.remap(gray_right, map1_r, map2_r, interpolation)
    cv2.imshow('Rectified Left Target Image', rectified_image_left)
    cv2.waitKey(0)
    cv2.imshow('Rectified Right Target Image', rectified_image_right)
    cv2.waitKey(0)

    # stereo block matching
    matcher = cv2.StereoBM_create(numDisparities = 256,blockSize = 15)
    disparity = matcher.compute(rectified_image_left, rectified_image_right)
    map = ( disparity - np.min(disparity) ) / ( np.max(disparity) - np.min(disparity) )
    cv2.imshow('disparity', map)
    cv2.waitKey(0)

        # stereo semi global block matching
        matcher = cv2.StereoSGBM_create(minDisparity = 10, numDisparities=256, blockSize=22)
        disparity = matcher.compute(rectified_image_left, rectified_image_right)
        map = ( disparity - np.min(disparity) ) / ( np.max(disparity) - np.min(disparity) )
        cv2.imshow('disparity', map)
        cv2.waitKey(0)

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
