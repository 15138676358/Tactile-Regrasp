import function
import matplotlib.pyplot as plt
import numpy as np
import os

def test_cylin_and_rect():
    """
    The cylinder and rectangle object only collect one finger, so the K matrix needs modified.
    To test the cylinder, i is range(12), and the file is {i*2}mm. The K2 and s2 need to be negative.
    To test the rectangle, i is range(10), and the file is {i*1}deg. The K2 is the same as K1.
    To test the pivoting, i is range(10), and the file is {i*10}mm. The K2 is the same as K1.
    To test the tangential vector field, use T_field to calculate the K matrix.
    """
    Ks, ss = [], []
    for i in range(0,10):
        # Load the data from the .npy files
        P_field = np.load(f'./temp_data/single_finger/pivoting/P_field_{i*10}mm.npy')
        F_field = np.load(f'./temp_data/single_finger/pivoting/F_field_{i*10}mm.npy')

        # Calculate scores
        F_mask = np.linalg.norm(F_field, axis=1) > 0.04
        N_field = function.calculate_normal(P_field)
        T_field = function.calculate_tangential(N_field, F_field)
        Kn1, sn1 = function.fit_K(N_field, F_mask, save_img=False)
        Kt1, st1 = function.fit_K(T_field, F_mask, save_img=False)
        Kn2, sn2 = function.fit_K(N_field, F_mask, save_img=False)
        Kt2, st2 = function.fit_K(T_field, F_mask, save_img=False)
        ss.append(sn1)
        # # for rectangle
        # K1, K2 = Kn1, Kn2
        # s1, s2 = np.array([sn1[0], sn1[1], sn1[2] - 1]), np.array([sn2[0], sn2[1], sn2[2] - 1])
        # # for cylinder
        # K1, K2 = Kn1, Kn2
        # K2[0, 1], K2[1, 0], s1, s2 = -K2[0, 1], -K2[1, 0], np.array([sn1[0], sn1[1], sn1[2] - 1]), np.array([sn2[0], -sn2[1], sn2[2] - 1])
        # for pivoting
        K1, K2 = Kt1, Kt2
        K2[0, 1], K2[1, 0], s1, s2 = -K2[0, 1], -K2[1, 0], np.array([st1[0] - 1, st1[1], st1[2]]), np.array([st2[0] - 1, -st2[1], st2[2]])

        J1, J2 = function.calculate_Jacobi(d=20)
        w = function.calculate_weight(d=20)
        delta_u = function.optimize_motion(K1, K2, J1, J2, s1, s2, 0.1*w)
        Ks.append(delta_u)


    plt.plot(Ks)
    plt.legend(['δyt', 'δzt', 'δθyt', 'δθzt'])
    plt.show()

def test_pivot():
    """
    The pivot situation have two fingers, so the K matrix can be calculated directly.
    """
    Ks, ss = [], []
    for i in range(10):
        # Load the data from the .npy files
        P_field = np.load(f'./temp_data/two_finger/pivoting/P_field_{i*1}deg.npy')
        F_field = np.load(f'./temp_data/two_finger/pivoting/F_field_{i*1}deg.npy')

        # Calculate scores
        F_mask = np.linalg.norm(F_field, axis=1) > 0.04
        N_field = function.calculate_normal(P_field)
        T_field = function.calculate_tangential(N_field, F_field)
        Kn1, sn1 = function.fit_K(N_field, F_mask, save_img=False)
        Kt1, st1 = function.fit_K(T_field, F_mask, save_img=False)
        Kn2, sn2 = function.fit_K(N_field, F_mask, save_img=False)
        Kt2, st2 = function.fit_K(T_field, F_mask, save_img=False)

        J_1, J_2 = function.calculate_Jacobi(d=20)
        w = function.calculate_weight(d=20)
        delta_un = function.optimize_motion(Kn1, Kn2, J_1, J_2, sn1, sn2, 0.01*w)
        delta_ut = function.optimize_motion(Kt1, Kt2, J_1, J_2, st1, st2, 0.01*w)
        delta_u = 0.9 * delta_un + 0.1 * delta_ut
        Ks.append(delta_u)
        ss.append(sn1)

    plt.plot(Ks)
    plt.legend(['δy', 'δz', 'δθy', 'δθz'])
    plt.show()

if __name__ == '__main__':
    test_cylin_and_rect()