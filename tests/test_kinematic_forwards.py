import os
import sys

sys.path.append(os.path.join(*1*[os.pardir]))
from kinematics import R_x, R_y, R_z, rotation_matrix_from_rotation_vector, \
                       homogenous_transformation, UrdfForwardKinematic, euler_from_rotation_matrix
from helper.pybullet_helper import get_joint_order
from simulation import Simulation, FrankaPanda

from scipy import spatial
import numpy as np
import tensorflow as tf
import pybullet as p


def test_R():
    R_fns = [R_x, R_y, R_z]
    vecs = [np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])]
    for fn, vec in zip(R_fns, vecs):
        for _ in range(10):
            # dimension and batch sizes
            batch_size = np.random.randint(low=1, high=9, size=None)
            
            # data sampling
            angle = np.random.uniform(low=0, high=2*np.pi, size=batch_size)

            # ground-truth
            with_scipy = [spatial.transform.Rotation.from_rotvec(rotvec=a*vec).as_matrix()
                         for a in angle]
            with_scipy = np.array(with_scipy).reshape([batch_size, 3, 3])

            # tensorflow-implemtation (to be tested)
            with_tf = fn(tf.reshape(angle.astype(np.float32), shape=[batch_size, 1]))
            with_tf = with_tf.numpy()

            assert np.max(with_scipy - with_tf) <= 1e-6
            assert with_tf.shape == (batch_size, 3, 3)

def test_homogenous_transformation():
    for _ in range(10):
        # dimension and batch sizes
        batch_size = np.random.randint(low=1, high=9, size=None)
        
        # data sampling (ground truth)
        R = np.random.uniform(low=0, high=99, size=[batch_size, 3,3])
        t = np.random.uniform(low=0, high=99, size=[batch_size, 3,1])
        Rt = np.concatenate([R,t], axis=-1)
        bottom = np.broadcast_to([[0,0,0,1]], shape=[batch_size, 1,4])
        T = np.concatenate([Rt, bottom], axis=-2)

        # tensorflow implementation (to be tested)
        T_tf = homogenous_transformation(
            R = tf.constant(R, dtype=tf.float32),
            t = tf.constant(t.reshape([batch_size, 3]), dtype=tf.float32),
        )
        T_tf = T_tf.numpy()

        assert np.max(T - T_tf) <= 1e-5
        assert T_tf.shape == (batch_size, 4, 4)

def test_rotation_matrix_from_rotation_vector():
    for _ in range(10):
        # dimension and batch sizes
        batch_size = np.random.randint(low=1, high=9, size=None)
        
        # data sampling
        vec_unnormalized = np.random.uniform(size=[batch_size, 3])
        vec_norm = np.linalg.norm(vec_unnormalized, axis=-1, keepdims=True)
        vec = vec_unnormalized / vec_norm
        angle = np.random.uniform(low=0, high=2*np.pi, size=[batch_size])

        # ground-truth
        with_scipy = [spatial.transform.Rotation.from_rotvec(rotvec=a*v).as_matrix()
                    for v, a in zip(vec, angle)]
        with_scipy = np.array(with_scipy).reshape([batch_size, 3, 3])

        # tensorflow-implemtation (to be tested)
        with_tf = rotation_matrix_from_rotation_vector(
            vec = tf.reshape(vec.astype(np.float32), [batch_size, 3]),
            angle = tf.reshape(angle.astype(np.float32), [batch_size])
        )
        with_tf = with_tf.numpy()

        assert np.max(with_scipy - with_tf) <= 1e-6
        assert with_tf.shape == (batch_size, 3, 3)

def test_euler_from_rotation_matrix():
    for _ in range(100):
        # dimension and batch sizes
        n_batch_dims = 1
        batch_shape = np.random.randint(low=1, high=9, size=n_batch_dims)
        n = np.prod(batch_shape)

        # data sampling (ground truth)
        eulers = np.random.uniform(low=0, high=2*np.pi, size=[*batch_shape, 3])
        R = [spatial.transform.Rotation.from_euler(seq='xyz', angles=e).as_matrix() for e in eulers.reshape([-1, 3])]
        R = np.array(R, dtype=np.float32).reshape([*batch_shape, 3,3])

        # tensorflow-implementation (to be tested)
        eulers_tf = euler_from_rotation_matrix(rotation_matrix=R)
        eulers_tf = eulers_tf.numpy()
        R_tf = [spatial.transform.Rotation.from_euler(seq='xyz', angles=e).as_matrix() for e in eulers_tf.reshape([-1, 3])]
        R_tf = np.array(R_tf, dtype=np.float32).reshape([*batch_shape, 3,3])

        assert np.max(R - R_tf) < 1e-4
        assert eulers_tf.shape == (*batch_shape, 3)

def test_taskmap_extraction_from_urdf():
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'urdf', 'franka_panda', 'panda.urdf')
 
    simulation = Simulation()
    simulation.connect()

    robot = FrankaPanda()
    simulation.populate_scene(robot)

    order = get_joint_order(robot.id, motor_joints_only=True)
    fkine = UrdfForwardKinematic(urdf_filepath=path, order=order)

    joint_names = [p.getJointInfo(bodyUniqueId=robot.id, jointIndex=i, physicsClientId=simulation.client_id)[1].decode('ascii')
                   for i in range(p.getNumJoints(bodyUniqueId=robot.id, physicsClientId=simulation.client_id))]
    
    for _ in range(1000):
        robot.q = np.random.uniform(low=robot.q_lim_low, high=robot.q_lim_high)[robot.idx_controllable]
        
        for i, joint_name in enumerate(joint_names):
            # pybullet
            link_state = p.getLinkState(bodyUniqueId=robot.id, linkIndex=i)
            pos, orn = link_state[4], link_state[5]
            R = spatial.transform.Rotation.from_quat(orn).as_matrix()
            T_pb = np.vstack([np.hstack([R, np.array(pos).reshape(-1, 1)]), np.array([[0,0,0,1]])])

            # tensorflow
            T_tf = fkine.forward(q=tf.constant(robot.q, dtype=tf.float32),
                                 frame=tf.constant(joint_name))
            
            assert np.max(np.abs(T_tf - T_pb)) < 1e-6


if __name__ == '__main__':
    test_R()
    test_rotation_matrix_from_rotation_vector()
    test_homogenous_transformation()
    test_taskmap_extraction_from_urdf()
    test_euler_from_rotation_matrix()
    print('Done!')