import numpy as np
import cv2

from .visualization import show_images
from .io import load_intrinsic_matrix

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

def match_keypoints(ds1, ds2, img1, img2, kp1, kp2, draw = False):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ds1, ds2, k=2)
    draw_matches = [[0, 0] for _ in range(len(matches))]
    point_matches = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            if draw:
                draw_matches[i] = [1, 0]
            point_matches.append(m)
    
    points1 = []
    points2 = []

    for match in point_matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        points1.append([x1, y1, 1.0])
        points2.append([x2, y2, 1.0])
    points1 = np.array(points1)
    points2 = np.array(points2)

    if len(draw_matches) > 0 and draw:
        matched = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches1to2=matches, outImg=None, matchColor=(255, 0, 0), singlePointColor=(0, 255, 255), matchesMask=draw_matches, flags =0)
    print(points2.shape, points1.shape)
    return (points1, points2)

def form_A_row(xr_row, xl_row):
    temp = np.zeros(9)
    temp[0] = xr_row[0] * xl_row[0]
    temp[1] = xr_row[1] * xl_row[0]
    temp[2] = xl_row[0]
    temp[3] = xr_row[0] * xl_row[1]
    temp[4] = xr_row[1] * xl_row[1]
    temp[5] = xl_row[1]
    temp[6] = xr_row[0]
    temp[7] = xr_row[1]
    temp[8] = 1
    return temp


def create_A_matrix(x_r, x_l, K_inv):
    """
    create the A matrix we will decompose via SVD. should be shape (Nx9) where N is the number of samples from SIFT

    N needs to be greater than 8 to make problem well-defined (N < 8 infinite number of solutions, cant find closed form solution?)

    [uu', vu', u', uv', vv', v', u, v, 1] is each row
    
    input comes in the form Nx3 for both xr and xl. each row of xr is [u, v, 1]. each row of xl is [u', v', 1]

    make sure to invert each 2d point into 3d ray direction again since we are doing essential matrix derivation

    """

    assert x_r.shape[0] == x_l.shape[0] #quick check that we are passing in same number of rows

    N = x_r.shape[0]

    A = np.zeros((N, 9))

    for i, _ in enumerate(x_r):
        A[i] = form_A_row(K_inv @ x_r[i], K_inv @ x_l[i]) #apply back projection via K^-1 (2d -> 3d ray)

    return A
    
def form_epipolar_constraint(x_r, x_l, K=None):
    """
    forms epipolar constraint. x_r^TEx_l=0

    used to estimate essential matrix E, which encodes rotation and translation between two camera frames.

    x_r and x_l are homogeneous points gotten from SIFT + FLANN pairs. we need to back project via K^-1 (camera intrinsic matrix inverse)

    geometrically, inverting the intrinsic matrix converts 2d image plane points to a 3d world ray direction (important that we know it is a direction vector, not a point), without K^-1 we are estimating F not E
    
    sift pairs should be shaped Nx9 where each row is one pair expanded    

    returns essential matrix (3x3)

    """
    if K is None:
        K = load_intrinsic_matrix()
    K_inv = np.linalg.inv(K)

    A = create_A_matrix(x_r, x_l, K_inv)
    print(A, A.shape)

    U, sigma, V_t = np.linalg.svd(A)

    print(U.shape, sigma.shape, V_t.shape)

    e = V_t[:, -1]

    E = e.reshape((3, 3))

    U, sigma, V_t = np.linalg.svd(E) #perform svd again to get R and t
    print(f"sigma matrix from E decomposed: 3rd singular value may not equal 0: {sigma}")
    sigma_fixed = np.diag([1, 1, 0]) #enforces rank 2 if we replace sigma with this, 
    E_clean = U @ sigma_fixed @ V_t 

    U, S, Vt = np.linalg.svd(E_clean)

    #we'll need to do point normalization for stability? clamping values bvetween 0 and 1 i guess

    return E


















    
