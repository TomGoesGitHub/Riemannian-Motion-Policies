import os
import sys

from scipy import spatial
import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*1*[os.pardir]))
from rmp import RiemannianMotionPolicy
from kinematics import UrdfForwardKinematic, euler_from_rotation_matrix, get_H_forEulerXYZ
from helper.pybullet_helper import get_joint_order
from simulation import Simulation, FrankaPanda

# class SimpleRmp(RiemannianMotionPolicy):
#     def _taskmap(self, q):
#         '''Task-map'''
#         T = self.mapping(q)
#         pos = T[..., :3, 3]
#         eulers = euler_from_rotation_matrix(rotation_matrix=T[..., :3,:3])
#         x = tf.concat([pos, eulers], axis=-1)
#         return x

def test_jacobians():
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'urdf', 'franka_panda', 'panda.urdf')
 
    simulation = Simulation()
    simulation.connect()

    robot = FrankaPanda()
    simulation.populate_scene(robot)

    order = get_joint_order(robot.id, motor_joints_only=True)
    fkine = UrdfForwardKinematic(urdf_filepath=path, order=order)

    joint_names = [p.getJointInfo(bodyUniqueId=robot.id, jointIndex=i)[1].decode('ascii')
                   for i in range(p.getNumJoints(bodyUniqueId=robot.id))]
    

    for _ in range(100):
        robot.q = np.random.uniform(low=robot.q_lim_low, high=robot.q_lim_high)[robot.idx_controllable]
        robot.qd = np.random.uniform(low=len(robot.q)*[-0.1], high=len(robot.q)*[0.1])

        
        for i, joint_name in enumerate(joint_names):
            # pybullet (ground truth)
            link_state = p.getLinkState(bodyUniqueId=robot.id, linkIndex=i)
            pos, orn = link_state[4], link_state[5]
            orn_euler = spatial.transform.Rotation.from_quat(orn).as_euler('xyz')
            R = spatial.transform.Rotation.from_quat(orn).as_matrix()
            T_pb = np.vstack([np.hstack([R, np.array(pos).reshape(-1, 1)]), np.array([[0,0,0,1]])])

            zero_vec = [0] * len(robot.q)
            J_trans_pb, J_rot_pb_geom = p.calculateJacobian(bodyUniqueId=robot.id, linkIndex=i, localPosition=[0,0,0],
                                                            objPositions=robot.q.tolist(), objVelocities=zero_vec,
                                                            objAccelerations=zero_vec)
            J_trans_pb, J_rot_pb_geom = np.array(J_trans_pb), np.array(J_rot_pb_geom)
            
            H = get_H_forEulerXYZ(orn_euler)
            J_rot_pb = np.matmul(np.linalg.inv(H), J_rot_pb_geom) # get analytical Jacobian from geometric Jacobian

            xd_pb = J_trans_pb @ robot.qd
            # tensorflow implementation
            x, _xd, J, _ = fkine.differentiate(
                    q = tf.constant(robot.q, dtype=tf.float32),
                    qd = tf.constant(robot.qd, dtype=tf.float32),
                    frame = tf.constant(joint_name)
            )

            J_trans_tf = tf.gather(J, indices=[3,7,11], axis=-2)
            xd = tf.gather(_xd, indices=[3,7,11], axis=-1)

            assert np.max(np.abs(J_trans_pb - J_trans_tf)) < 1e-6
            assert np.max(np.abs(xd_pb - xd)) <= 1e-6


if __name__ == '__main__':
    test_jacobians()
    print('Done!')