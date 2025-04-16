import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d as o3d

"""
This module contains functions for tactile field processing and visualization.
It includes functions to initialize the tactile sensor, draw the tactile field,
fit a plane to the tactile data, calculate the Jacobian matrix, and optimize motion.
Definitions:
    - Tactile field: A representation of the tactile sensor data, including position, force, and vector fields.
        - P_field, F_field, N_field, T_field
    - F_mask: A mask indicating the valid contact regions of the tactile field.
    - K: A matrix including the coefficient of planar fitting of tactile field.
    - s: The mean value of the tactile field.
    - J: The Jacobian matrix representing the transformation from hand motion to finger motion.
"""
def draw_tactile(F_field, F_mask):
    """
    Draws the tactile field using quiver plot.
    Args:
        F_field (np.ndarray): The tactile field data.
        F_mask (np.ndarray): The mask for the tactile field.
    Returns:
        Saves the tactile field image as 'tactile_image.png'.
    """
    X, Y = np.meshgrid(np.arange(20), np.arange(20))
    Fx, Fy, Fz, F_mask = F_field[:, 0].reshape(20, 20), F_field[:, 1].reshape(20, 20), F_field[:, 2].reshape(20, 20), F_mask.reshape(20, 20)
    plt.quiver(X, Y, Fx * F_mask, Fy * F_mask, Fz * F_mask, cmap='Blues', pivot='tail', scale=10, width=0.005, headwidth=4, headlength=6, headaxislength=4)
    plt.axis('equal')
    plt.colorbar()
    plt.savefig('tactile_image.png')
    plt.close()

def calculate_normal(P_field):
    """
    Calculates the normal vector field from the tactile sensor data.
    Args:
        P_field (np.ndarray): The tactile sensor data.
        knn (int): The number of nearest neighbors to use for normal estimation.
    Returns:
        N_field (np.ndarray): The normal vector field.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_field)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=8))
    N_field = np.asarray(pcd.normals)
    # edge points may return negative nz, use negative_mask to convert the normal vectors of edge points.
    negative_mask = N_field[:, 2] < 0
    N_field[negative_mask] = -N_field[negative_mask]

    return N_field

def calculate_tangential(N_field, F_field):
    """Calculate the tangential vector from normal and force vectors.
    Args: 
        N_field (np.ndarray): Normal vector field. 
        F_field (np.ndarray): Force field.
    Returns: 
        Tangential vector field.
    """
    Fn_field = np.sum(N_field * F_field, axis=1)[:, np.newaxis] * N_field
    Ft_field = F_field - Fn_field
    T_field = Ft_field / np.linalg.norm(Ft_field, axis=1)[:, np.newaxis]
    
    return T_field

def fit_K(N_field, F_mask, save_img=False):
    """
    Fits the tactile vector field to a plane using least squares method.
    Args:
        N_field (np.ndarray): The vector field data.
        F_mask (np.ndarray): The mask for the tactile field.
        save_img (bool): Whether to save the fitted image or not.
    Returns:
        K (np.ndarray): The matrix including the fitted plane coefficients.
        s (np.ndarray): The mean values of the tactile field.
        """
    X, Y = np.meshgrid(np.arange(20), np.arange(20))
    Nx, Ny, Nz, F_mask = N_field[:, 0].reshape(20, 20), N_field[:, 1].reshape(20, 20), N_field[:, 2].reshape(20, 20), F_mask.reshape(20, 20)
    X, Y, Nx, Ny, Nz = X[F_mask], Y[F_mask], Nx[F_mask], Ny[F_mask], Nz[F_mask]
    fit_input = np.column_stack((X.reshape(-1), Y.reshape(-1), np.ones_like(X.reshape(-1))))
    coeffx, residualx, _, _  = np.linalg.lstsq(fit_input, Nx.reshape(-1), rcond=None)
    coeffy, residualy, _, _  = np.linalg.lstsq(fit_input, Ny.reshape(-1), rcond=None)
    sx, sy = np.mean(Nx), np.mean(Ny)
    
    if save_img == True:
        fig = plt.figure(figsize=(10, 6))
        surfx, surfy = np.meshgrid(range(X.min() - 2, X.max() + 2), range(Y.min() - 2, Y.max() + 2))
        
        surfz = coeffx[0] * surfx + coeffx[1] * surfy + coeffx[2]
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X, Y, Nx, c=Nx, cmap='viridis', alpha=0.8)
        ax.plot_surface(surfx, surfy, surfz, alpha=0.3, color='r')
        ax.set_title(f's^x: {sx:.2f} \n coeff*100: {[round(c*100, 2) for c in coeffx]} \n residual*100: {residualx[0]*100:.2f}', fontsize=12)
        
        surfz = coeffy[0] * surfx + coeffy[1] * surfy + coeffy[2]
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(X, Y, Ny, c=Ny, cmap='viridis', alpha=0.8)
        ax.plot_surface(surfx, surfy, surfz, alpha=0.3, color='r')
        ax.set_title(f's^y: {sy:.2f} \n coeff*100: {[round(c*100, 2) for c in coeffy]} \n residual*100: {residualy[0]*100:.2f}', fontsize=12)
        
        plt.savefig('fit_image.png')
        plt.close()

    return np.array([[coeffx[0], coeffx[1], 0, 1], [coeffy[0], coeffy[1], -1, 0]]), np.array([sx, sy])

def calculate_Jacobi(d):
    """
    Calculates the Jacobian matrix of the fingers, derived under the assumption of two-finger gripper.
    Args:
        d (float): The distance between the two fingers.
    Returns:
        J_i (np.ndarray): The Jacobian matrix for the fingers.
    """
    J_1 = np.array([[0, 1, d, 0],[-1, 0, 0, d],[0, 0, 0, 1],[0, 0, -1, 0]])
    J_2 = np.array([[0, 1, -d, 0],[1, 0, 0, d],[0, 0, 0, 1],[0, 0, 1, 0]])

    return J_1, J_2

def calculate_weight(d):
    """
    Calculates the weight matrix for the optimization process, scaling the rotation to the translation with a distance variable.
    Args:
        d (float): The distance between the two fingers.
    Returns:
        w (np.ndarray): The weight matrix.
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0] ,[0, 0, d, 0] ,[0, 0, 0, d]])

def optimize_motion(K_1, K_2, J_1, J_2, s_1, s_2, w):
    """
    Optimizes the motion of the fingers using the Jacobian matrices and the tactile field data by solving the linear problem.
    Args:
        K_i (np.ndarray): The matrix including the coefficients of the fingers.
        J_i (np.ndarray): The Jacobian matrix for the fingers.
        s_i (np.ndarray): The mean values of the tactile fields.        
        w (np.ndarray): The weight matrix.
    Returns:
        delta_u (np.ndarray): The optimized motion of the fingers.
    """
    KJ_1, KJ_2 = K_1 @ J_1, K_2 @ J_2
    A = KJ_1.T @ KJ_1 + KJ_2.T @ KJ_2 + w.T @ w
    b = KJ_1.T @ s_1 + KJ_2.T @ s_2
    delta_u = np.linalg.solve(A, b)

    return delta_u


