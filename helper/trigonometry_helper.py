import pybullet as p
from scipy.spatial.transform import Rotation
import numpy as np


# def get_H_fromQuaternions(quaternions): 
#     '''
#     Returns the Matrix H which maps the angular vecolity with respect to the Frame-0 to the time derivative of the quaternions
#     (compare Woernle 2016 - Mehrkörpersysteme Eq. 3.211)
#     note: The order of the quaternions is adapted to the PyBullet-Representation
#     '''
#     p_s = quaternions[3]
#     p_vec = np.array(quaternions[:3])
#     p_tilde = np.cross(p_vec, np.identity(p_vec.shape[0]) * -1) # p_tilde is the cross product matrix for p_vec
#     H = np.concatenate([p_s * np.eye(p_vec.shape[0]) - p_tilde, -p_vec], axis=0)
#     return H

def get_H_forEulerXYZ(eulers):
    '''
    Returns the Matrix H, which maps the angular vecolity with respect to the Frame-0 to the time derivative of the euler angles.
    
        gamma_dot_vec = H * omega
        with gamma_dot_vec = [alpha_dot, beta_dot, gamma_dot]


    Some Clarifikation about the euler-angle convention:
        1) xyz = xyz extrensic = zy'x'' intrinsic
        2) xyz extrinsic is used by pybullet and tensorflow
    '''
    alpha, beta, gamma = eulers
    s_beta = np.sin(beta)
    c_beta = np.cos(beta)
    s_gamma = np.sin(gamma)
    c_gamma = np.cos(gamma)
    H = np.array([[c_beta*c_gamma, -s_gamma, 0],
                  [c_beta*s_gamma,  c_gamma, 0],
                  [-s_beta,               0, 1]])
    return H

def getSO3FromEulerXYZ(orn_euler_xyz):
    psi, theta, phi = orn_euler_xyz
    so3 = sm.SO3.Rz(phi) * sm.SO3.Ry(theta) * sm.SO3.Rx(psi)
    return so3.A

def getSO3FromQuaternions(orn_quat):
    orn_euler_xyz = p.getEulerFromQuaternion(orn_quat)
    so3 = getSO3FromEulerXYZ(orn_euler_xyz)
    return so3

def getQuaternionFromSE3(se3):
    return getQuaternionFromSO3(se3.R)

def getQuaternionFromSO3(so3):
    '''
    Implements the Algorithm of Shepperd, which returns quaternions for a given Rotation Matrix. 
    (see Woernle 2016 - Mehrkoerpersysteme, Chapter 3.7 for details)
    '''
    
    # if type(so3) == sm.SO3: # todo: remove
    #     T = so3.A
    # else:
    
    T = so3
    trace = np.trace(T) 
    
    # get Matrix indices for clarification and correspondance with Woernle 2016
    T_11, T_12, T_13 = T[0, :] #T_11, T_22, T_33 = np.diag(T)
    T_21, T_22, T_23 = T[1, :]
    T_31, T_32, T_33 = T[2, :]

    k = np.argmax(np.array([trace, T_11, T_22, T_33]))

    if k == 0:
        T_kk = trace
        p_s = 0.5 * np.sqrt(1 + 2*T_kk - trace) 
        p_x = (T_32 - T_23) / (4 * p_s)
        p_y = (T_13 - T_31) / (4 * p_s)
        p_z = (T_21 - T_12) / (4 * p_s)

    if k == 1:
        T_kk = T_11
        p_x = 0.5 * np.sqrt(1 + 2*T_kk - trace)

        p_s = (T_32 - T_23) / (4 * p_x)
        p_y = (T_21 + T_12) / (4 * p_x)
        p_z = (T_13 + T_31) / (4 * p_x)

    if k == 2:
        T_kk = T_22
        p_y = 0.5 * np.sqrt(1 + 2*T_kk - trace)

        p_s = (T_13 - T_31) / (4 * p_y)
        p_x = (T_21 + T_12) / (4 * p_y)
        p_z = (T_32 + T_23) / (4 * p_y)

    if k == 3:
        T_kk = T_33
        p_z = 0.5 * np.sqrt(1 + 2*T_kk - trace)

        p_s = (T_21 - T_12) / (4 * p_z)
        p_x = (T_13 + T_31) / (4 * p_z)
        p_y = (T_32 + T_23) / (4 * p_z)
  
    s = -1 if p_s < 0 else 1 #sign
    return [s*p_x, s*p_y, s*p_z, s*p_s]


def get_array(*args):
    result = []
    for arg in args:
        result.append( np.array(arg) )
    return result