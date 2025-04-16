import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import sys
import shutil
import subprocess
import time
sys.path.append('/home/a01/Workspace/a01_franka_ws/src/Grasp_wy')
sys.path.append('/home/a01/Workspace/a01_franka_ws/src/Grasp_wy/Grasp_Failure_Recovery')
sys.path.append('/home/a01/Workspace/a01_franka_ws/src/Grasp_wy/Grasp_API')
import Grasp_Failure_Recovery.graspagents as graspagents
import Grasp_API.control as control
import Grasp_API.core as core

def initialize():
    tac3d = control.Tac3D_Client()
    hand = control.Dexhand_Client()
    # arm = control.Arm_Client()
    tac3d.start_sensor()
    hand.reset_actuator()
    # arm.reset_actuator()
    # agent = graspagents.GraspAgent_bayes(config={'env': 'GraspEnv_v3'})
    # agent = graspagents.GraspAgent_rl({'env': 'GraspEnv_v3', 'model': 'SAC'})

    return tac3d, hand, 1

def draw_tactile(F_field, F_mask):
    X, Y = np.meshgrid(np.arange(20), np.arange(20))
    Fx, Fy, Fz = F_field[:, 0].reshape(20, 20), F_field[:, 1].reshape(20, 20), F_field[:, 2].reshape(20, 20)
    F_mask = F_mask.reshape(20, 20)
    # Fx, Fy = 0.1 * Fx / (Fz + 0.01 * (Fz < 0.01)), 0.1 * Fy / (Fz + 0.01 * (Fz < 0.01))
    image = plt.quiver(X, Y, Fx * F_mask, Fy * F_mask, Fz * F_mask, cmap='Blues', pivot='tail', scale=10, width=0.005, headwidth=4, headlength=6, headaxislength=4)
    plt.axis('equal')
    plt.colorbar()
    plt.savefig('tactile_image.png')
    plt.close()

def fit_K(N_field, F_mask, save_img=False):
    X, Y = np.meshgrid(np.arange(20), np.arange(20))
    Nx, Ny, Nz, F_mask = N_field[:, 0].reshape(20, 20), N_field[:, 1].reshape(20, 20), N_field[:, 2].reshape(20, 20), F_mask.reshape(20, 20)
    X, Y, Nx, Ny, Nz = X[F_mask], Y[F_mask], Nx[F_mask], Ny[F_mask], Nz[F_mask]
    fit_input = np.column_stack((X.reshape(-1), Y.reshape(-1), np.ones_like(X.reshape(-1))))
    coeffx = np.linalg.lstsq(fit_input, Nx.reshape(-1), rcond=None)[0]
    coeffy = np.linalg.lstsq(fit_input, Ny.reshape(-1), rcond=None)[0]
    sx, sy = np.mean(Nx), np.mean(Ny)
    
    if save_img == True:
        fig = plt.figure(figsize=(10, 6))
        surfx, surfy = np.meshgrid(range(X.min() - 2, X.max() + 2), range(Y.min() - 2, Y.max() + 2))
        
        surfz = coeffx[0] * surfx + coeffx[1] * surfy + coeffx[2]
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X, Y, Nx, c=Nx, cmap='viridis', alpha=0.8)
        ax.plot_surface(surfx, surfy, surfz, alpha=0.3, color='r')
        ax.set_title(f's^x: {sx}')
        
        surfz = coeffy[0] * surfx + coeffy[1] * surfy + coeffy[2]
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(X, Y, Ny, c=Ny, cmap='viridis', alpha=0.8)
        ax.plot_surface(surfx, surfy, surfz, alpha=0.3, color='r')
        ax.set_title(f's^y: {sy}')
        
        plt.savefig('fit_image.png')
        plt.close()

    return np.array([[coeffx[0], coeffx[1], 0, 1], [coeffy[0], coeffy[1], -1, 0]]), np.array([sx, sy])

def calculate_Jacobi(d):
    J_1 = np.array([[0, 1, d, 0],[-1, 0, 0, d],[0, 0, 0, 1],[0, 0, -1, 0]])
    J_2 = np.array([[0, 1, -d, 0],[1, 0, 0, d],[0, 0, 0, 1],[0, 0, 1, 0]])

    return J_1, J_2

def calculate_weight(d):
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0] ,[0, 0, d, 0] ,[0, 0, 0, d]])

def optimize_motion(K_1, K_2, J_1, J_2, s_1, s_2, w):
    KJ_1, KJ_2 = K_1 @ J_1, K_2 @ J_2
    A = KJ_1.T @ KJ_1 + KJ_2.T @ KJ_2 + w.T @ w
    b = KJ_1.T @ s_1 + KJ_2.T @ s_2
    delta_u = np.linalg.solve(A, b)

    return delta_u


def calculate_normal(P_field):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_field)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=8))
    N_field = np.asarray(pcd.normals)
    # edge points may return negative nz, use mask to convert the normal vectors of edge points.
    negative_mask = N_field[:, 2] < 0
    N_field[negative_mask] = -N_field[negative_mask]

    return N_field

if __name__ == '__main__':
    tac3d, hand, arm = initialize()
    hand.grasp_pose(20.0)
    hand.grasp_force(4.0)


    tac3d.gate = True
    time.sleep(1.0)
    tac3d.gate = False
    YA_019, YA_020 = tac3d.get_last_frame()
    tac3d.clear_frames()

    for i, YA in zip([0, 1], [YA_019, YA_020]):
        F_field = np.array(YA['F_field'])[::-1, :]  # single side tactile sensor, don't need SN check, modify the order
        P_field = np.array(YA['P_field'])[::-1, :]
        # F_mask = np.linalg.norm(F_field, axis=1) > 0.04
        # N_field = calculate_normal(P_field)
        # if i == 0:
        #     K_1, s_1 = fit_K(N_field, F_mask, save_img=True)
    #     if i == 1:
    #         K_2, s_2 = fit_K(N_field, F_mask, save_img=True)
    # J_1, J_2 = calculate_Jacobi(d=10)
    # # J_2 = J_1
    # w = calculate_weight(d=10)
    # delta_u = optimize_motion(K_1, K_2, J_1, J_2, s_1, s_2, 0.01*w)
    # print(delta_u)
    # draw_tactile(N_field, F_mask)
    np.save('./temp_data/F_field_9deg', F_field)
    np.save('./temp_data/P_field_9deg', P_field)


    hand.grasp_pose(0.0)
