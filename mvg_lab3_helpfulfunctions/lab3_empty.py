#!/usr/bin/env python3
"""
Lab 3 - Multi-View Geometry - Epipolar Geometry and Stereo
"""

import numpy as np
import matplotlib.pyplot as plt

from mvg_project_point_to_image_plane import project_points_to_image_plane
from mvg_show_projected_points import show_projected_points
from mvg_compute_epipolar_geom import compute_epipolar_geometry
from mvg_show_epipolar_lines import show_epipolar_lines
from mvg_show_epipoles import show_epipoles

def main():
    # Step 1 - Camera 1
    au1, av1, uo1, vo1 = 100, 120, 128, 128
    image_size = (256, 256)

    # Step 2 - Camera 2
    au2, av2, uo2, vo2 = 90, 110, 128, 128
    ax, by, cz = 0.1, np.pi/4, 0.2
    tx, ty, tz = -1000, 190, 230

    # Step 3 - Intrinsics and projection matrices
    K1 = np.array([[au1, 0, uo1], [0, av1, vo1], [0, 0, 1]], float)
    wR1c = np.eye(3)
    wt1c = np.zeros(3)

    # Note: ************** You have to add your own code from here onward ************
    K2 = np.array([[au2, 0, uo2], [0, av2, vo2], [0, 0, 1]])
    #wR2c = np.array([ax, by, cz])

    Rx = np.array([
    [1, 0,          0],
    [0, np.cos(ax), -np.sin(ax)],
    [0, np.sin(ax),  np.cos(ax)]
    ])

    Ry = np.array([
        [ np.cos(by), 0, np.sin(by)],
        [0,           1, 0],
        [-np.sin(by), 0, np.cos(by)]
    ])

    Rz = np.array([
        [np.cos(cz), -np.sin(cz), 0],
        [np.sin(cz),  np.cos(cz), 0],
        [0,           0,          1]
    ])

    wR2c = Rz @ Ry @ Rx
    wt2c = np.array([tx, ty, tz])

    P1 = K1 @ np.hstack([wR1c.T, -wR1c.T @ wt1c.reshape(3, 1)])
    P2 = K2 @ np.hstack([wR2c.T, -wR2c.T @ wt2c.reshape(3, 1)])
    
    #P1  # placeholder

    # Step 4 
    # Attention: This is an invented matrix just to have some input for the drawing functions. You have to compute it properly

    t_skew = np.array([
        [0, -tz, ty],
        [tz, 0, 0],
        [-ty, tx, 0]
    ])

    F = (
        np.linalg.inv(K2).T
        @ wR2c.T
        @ t_skew
        @ np.linalg.inv(K1)
    )

    '''
    F = np.array([[3e-05, 7e-05, -0.006],
                  [2e-05, -2e-05, 0.01],
                  [-0.009, -0.01, 1]], float)
    '''
    F /= F[2, 2]
    print("Step 4: Analytically obtained F:\n", F)

    # Step 5 - 3D points
    V = np.array([[100,300,500,700,900,100,300,500,700,900,100,300,500,700,900,100,300,500,700,900],
                  [-400,-400,-400,-400,-400,-40,-40,-40,-40,-40,40,40,40,40,40,400,400,400,400,400],
                  [2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000]], float)

    # Step 6 - Projection and visualization
    cam1_p2d = project_points_to_image_plane(V, P1)
    cam2_p2d = project_points_to_image_plane(V, P2)

    cam1_fig = show_projected_points(cam1_p2d, image_size, "Projected points on image plane 1")
    cam2_fig = show_projected_points(cam2_p2d, image_size, "Projected points on image plane 2")
    ax1, ax2 = cam1_fig.axes[0], cam2_fig.axes[0]

    _, _, c1, c2 = compute_epipolar_geometry(cam1_p2d, cam2_p2d, F)
    margins = ((-400, 300), (1, 400))
    show_epipolar_lines(ax1, ax2, c1, c2, margins, color='b')

    ep1, ep2 = np.array([-300,200,1]), np.array([200,50,1]) # These are dummy values for the epipoles just for illustrating. You have to compute them
    show_epipoles(ax1, ax2, ep1, ep2)

    plt.show()

if __name__ == "__main__":
    main()
