import numpy as np
import cv2

def calculate_keypoints(img1, img2, draw = False):
    if img1 is None or img2 is None:
        raise ValueError("one of the input images is None.")
    sift = cv2.SIFT_create()
    img1_keypoints, img1_descriptors = sift.detectAndCompute(img1, None)
    img2_keypoints, img2_descriptors = sift.detectAndCompute(img2, None)
    if draw:
        img1_kp_draw = cv2.drawKeypoints(img1, img1_keypoints, None)
        img2_kp_draw = cv2.drawKeypoints(img2, img2_keypoints, None)
        show_images(img1_kp_draw, img2_kp_draw)
    return img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors

def match_keypoints(ds1, ds2, img1, img2, kp1, kp2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ds1, ds2, k=2)
    
    good_matches = [[0, 0] for _ in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good_matches[i] = [1, 0]
    matched = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches1to2=matches, outImg=None, matchColor=(255, 0, 0), singlePointColor=(0, 255, 255), matchesMask=good_matches, flags =0)

def calibrate_camera(calibration_image):
    retval, corners = cv2.findChessboardCorners(calibration_image, )

def eightpoint():
    pass

