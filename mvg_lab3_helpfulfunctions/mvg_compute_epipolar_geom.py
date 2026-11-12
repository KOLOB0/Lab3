import numpy as np

def compute_epipolar_geometry(pts1, pts2, F):
    """
    Compute epipolar lines and line coefficients on each image.
    Inputs are 2xN or 3xN; we will homogenize if needed.
    Returns
    -------
    lm1, lm2 : (3,N) ndarray
        Epipolar lines on image 1 and image 2 (ax + by + c = 0).
    l_coef_1, l_coef_2 : (2,N) ndarray
        Slope-intercept form y = m x + q for each line in each image,
        with rows [m; q].
    """
    if pts1.shape[0] == 2:
        pts1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    if pts2.shape[0] == 2:
        pts2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
    lm2 = F @ pts1
    lm1 = F.T @ pts2
    # Convert to y = m x + q; beware division by zero on b
    eps = 1e-12
    b1 = np.where(np.abs(lm1[1]) < eps, np.sign(lm1[1]) * eps + eps, lm1[1])
    b2 = np.where(np.abs(lm2[1]) < eps, np.sign(lm2[1]) * eps + eps, lm2[1])
    m1 = -lm1[0] / b1
    q1 = -lm1[2] / b1
    m2 = -lm2[0] / b2
    q2 = -lm2[2] / b2
    l_coef_1 = np.vstack([m1, q1])
    l_coef_2 = np.vstack([m2, q2])
    return lm1, lm2, l_coef_1, l_coef_2
