import cv2
import numpy as np
import glob

IMAGE_WIDTH = 3280
IMAGE_HEIGHT= 2464
square_size = 1.9      # rect size [cm]
pattern_size = (10, 7)

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints_left = []
imgpoints_right = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# path of calibration images
PATH = "./calib_image_stereo/"
images_left = files =glob.glob(PATH + "*_left.jpg")
images_right = files =glob.glob(PATH + "*_right.jpg")

def main():
    # extract corners from chess board images
    for i, image_left in enumerate(images_left):
        # show filename 
        print(image_left)
        image_right = image_left.rstrip("_left.jpg") + "_right.jpg"
        print(image_right)

        # load images
        bgr_left = cv2.imread(image_left, cv2.IMREAD_COLOR)
        bgr_right = cv2.imread(image_right, cv2.IMREAD_COLOR)
        
        gray_left = cv2.cvtColor(bgr_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(bgr_right , cv2.COLOR_BGR2GRAY)

        # find corner from chessboard
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

        if ret_left == True and ret_right == True:
            print("corner detected!!!")
            corners2_left = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
            corners2_right = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)

            # Draw and display the corners
            crit = 5
            img_left = cv2.drawChessboardCorners(bgr_left, pattern_size, corners2_left,ret_left)
            img_right = cv2.drawChessboardCorners(bgr_right, pattern_size, corners2_right,ret_right)
            resized_img_left  = cv2.resize(img_left, (IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
            resized_img_right = cv2.resize(img_right,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
            # cv2.imshow(image_left, resized_img_left)
            # cv2.waitKey(0)
            # cv2.imshow(image_right, resized_img_right)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            imgpoints_left.append(corners_left.reshape(-1, 2))
            imgpoints_right.append(corners_right.reshape(-1, 2))
            objpoints.append(pattern_points)
        else:
            print("not detected")

    # stereo calibrate camera parameters
    ret, A1, D1, A2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints_left, imgpoints_right,None,None,None,None, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=0, criteria=criteria)

    if ret:
        # show parameters
        print("RMS = ", ret)
        print("A1 = \n", A1)
        print("A2 = \n", A2)
        print("D1 = \n", D1)
        print("D2 = \n", D2)
        print("R = \n", R)
        print("T = \n", T)

        # save camera parameters
        np.save(PATH + "R", R)
        np.save(PATH + "T", T)
        np.save(PATH + "A1", A1)
        np.save(PATH + "A2", A2)
        np.save(PATH + "D1", D1)
        np.save(PATH + "D2", D2)

    for i, image in enumerate(images_left):
        img_left = cv2.imread(image)        
        image_right = image.rstrip("_left.jpg") + "_right.jpg"
        img_right= cv2.imread(image_right)
        resultImg_left  = cv2.undistort(img_left, A1, D1, None)
        resultImg_right = cv2.undistort(img_right, A2, D2, None)

        # show undistorted images 
        resized_img_left  = cv2.resize(resultImg_left, (IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
        resized_img_right = cv2.resize(resultImg_right,(IMAGE_WIDTH//crit, IMAGE_HEIGHT//crit))
        cv2.imshow('undistorted left img', resized_img_left)
        cv2.waitKey(0)
        cv2.imshow('undistorted rightimg', resized_img_right )
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
