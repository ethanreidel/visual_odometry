import numpy as np
import cv2

from .visualization import show_images
from .io import load_intrinsic_matrix

def calculate_keypoints(img1, img2):
    if img1 is None or img2 is None:
        raise ValueError("one of the input images is None.")
    sift = cv2.SIFT_create()
    img1_keypoints, img1_descriptors = sift.detectAndCompute(img1, None)
    img2_keypoints, img2_descriptors = sift.detectAndCompute(img2, None)
    return img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors

def match_keypoints(ds1,ds2,img1,img2,kp1,kp2,draw=False,return_visualization=False,max_display_matches=None):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ds1, ds2, k=2)
    point_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
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

    matched = None
    if draw or return_visualization:
        displayed_matches = point_matches
        if max_display_matches is not None:
            if max_display_matches < 1:
                raise ValueError("max_display_matches must be greater than zero.")
            displayed_matches = sorted(point_matches, key=lambda match: match.distance)[:max_display_matches]

        matched = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        for index, match in enumerate(displayed_matches):
            previous_point = tuple(
                round(value) for value in kp1[match.queryIdx].pt
            )
            current_point = tuple(round(value) for value in kp2[match.trainIdx].pt)

            hue = int(((index * 0.61803398875) % 1.0) * 179)
            hsv_color = np.uint8([[[hue, 220, 255]]])
            color = tuple(
                int(value)
                for value in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
            )

            cv2.line(matched, previous_point, current_point, color, 1, cv2.LINE_AA)
            cv2.drawMarker(
                matched,
                previous_point,
                color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=7,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(matched, current_point, 3, color, -1, cv2.LINE_AA)
        if draw:
            show_images(matched)

    if return_visualization:
        return points1, points2, matched
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
    
def find_best_R_and_t(candidates, K_inv, x1, x2):
    """
    finds best R and t based on cheirality test (basically checking which points have Z>0 in both cameras)

    furthermore, we check Z in P1[2] ([X, Y, Z]) and P2 which equals RP2+t[2]

    point passes if Z1 > 0 and Z2 > 0
    
    data is a list of x1, x2 pairs
    """
    best_count = -1
    best_R, best_t = None, None
    for R, t in candidates:
        count = 0

        for p1, p2 in zip(x1, x2):
            ray1 = K_inv @ p1
            ray2 = K_inv @ p2

            #form system of equations to solve for lamb1/2 (depth)
            
            A = np.column_stack((R @ ray1, -ray2))
            b = -t

            lambdas, *_ = np.linalg.lstsq(A, b, rcond=None)
            lambda1, lambda2 = lambdas

            if lambda1 > 0 and lambda2 > 0:
                count+=1
        if count > best_count:
            best_count = count
            best_R = R
            best_t = t

    return best_R, best_t

def visual_odometry_pipeline(x_r, x_l, K=None):
    """
    forms epipolar constraint. x_r^TEx_l=0

    used to estimate essential matrix E, which encodes rotation and translation between two camera frames.

    x_r and x_l are homogeneous points gotten from SIFT + FLANN pairs. we need to back project via K^-1 (camera intrinsic matrix inverse)

    geometrically, inverting the intrinsic matrix converts 2d image plane points to a 3d world ray direction (important that we know it is a direction vector, not a point), without K^-1 we are estimating F not E
    
    sift pairs should be shaped Nx9 where each row is one pair expanded    
    """

    if K is None:
        K = load_intrinsic_matrix()
    K_inv = np.linalg.inv(K)

    A = create_A_matrix(x_r, x_l, K_inv)
    _, _, Vt = np.linalg.svd(A)

    e = Vt[-1] #take last column of Vt? directions in the input space of A? (space where xl lives)

    E = e.reshape((3, 3))

    U, _, Vt = np.linalg.svd(E) #perform svd again to get R and t
    sigma_fixed = np.diag([1, 1, 0]) #enforces rank 2 if we replace sigma with this to rebuild E cleanly
    E_clean = U @ sigma_fixed @ Vt

    U, _, Vt = np.linalg.svd(E_clean)

    #this part is slightly magic not going to lie, but i think it's dealing with the sign ambiguity when determining R
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    #U[:, 2] takes the 3rd column of U corresponding to the smallest singular vector
    #E^T U[:, 2] = 0, direction that E^T maps to zero.
    t1 = U[:, 2]
    t2 = -U[:, 2]

    candidates = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

    #we'll need to do point normalization for stability? clamping values bvetween 0 and 1 i guess

    #cheirality test: figure out which candidate pair actually shows up in the camera view
    #p x (PX) = 0. p is observed sift pixel in homogeneous form.
    #P is camera matrix built from R and t candidates
    #X is found via SVD
    best_R, best_t = find_best_R_and_t(candidates, K_inv, x_r, x_l)

    
    


















    
