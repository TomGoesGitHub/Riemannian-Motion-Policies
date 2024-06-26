import os
import sys
import pybullet as p
import tensorflow as tf
import numpy as np
from scipy import spatial

sys.path.append(os.path.join(*1*[os.pardir]))
from simulation import Simulation, FrankaPanda
from taskmap import TaskmapByFunction, TaskmapFrom4x4ToPosition, TaskmapFrom4x4ToEuler, chain_taskmaps
from kinematics import UrdfForwardKinematic
from helper.pybullet_helper import get_joint_order
from helper.trigonometry_helper import get_H_forEulerXYZ
from kinematics import euler_from_rotation_matrix # todo: tmp

path = os.path.join(os.path.dirname(__file__), os.pardir, 'urdf', 'franka_panda', 'panda.urdf')

def test_jacobians():
    simulation = Simulation()
    simulation.connect()

    robot = FrankaPanda()
    simulation.populate_scene(robot)

    order = get_joint_order(robot.id, motor_joints_only=True)
    fkine = UrdfForwardKinematic(urdf_filepath=path, order=order)

    joint_names = [p.getJointInfo(bodyUniqueId=robot.id, jointIndex=i)[1].decode('ascii')
                    for i in range(p.getNumJoints(bodyUniqueId=robot.id))]

    for i, joint_name in enumerate(joint_names):   
        # taskmaps
        taskmap_joint_space_to_endeffector_4x4 = TaskmapByFunction(
            forward_fn=lambda q: fkine.forward(q, frame=tf.constant(joint_name)),
            differentiate_fn=lambda q, qd: fkine.differentiate(q, qd, frame=tf.constant(joint_name))
        )

        taskmap_endeffector_4x4_to_pos = TaskmapFrom4x4ToPosition()
        taskmap_joint_space_to_pos = chain_taskmaps(taskmap_joint_space_to_endeffector_4x4,
                                                    taskmap_endeffector_4x4_to_pos)

        taskmap_endeffector_4x4_to_euler = TaskmapFrom4x4ToEuler()
        taskmap_joint_space_to_euler = chain_taskmaps(taskmap_joint_space_to_endeffector_4x4,
                                                    taskmap_endeffector_4x4_to_euler)

        for _ in range(50):
            robot.q = np.random.uniform(low=robot.q_lim_low, high=robot.q_lim_high)[robot.idx_controllable]
            
            # pybullet (ground truth)
            link_state = p.getLinkState(bodyUniqueId=robot.id, linkIndex=i)
            pos, orn = link_state[4], link_state[5]
            orn_euler = spatial.transform.Rotation.from_quat(orn).as_euler('xyz')
            R_pb = spatial.transform.Rotation.from_euler('xyz', orn_euler).as_matrix()

            zero_vec = [0] * len(robot.q)
            J_trans_pb, J_rot_pb_geom = p.calculateJacobian(bodyUniqueId=robot.id,
                                                            linkIndex=i,
                                                            localPosition=[0,0,0],
                                                            objPositions=robot.q.tolist(),
                                                            objVelocities=zero_vec,
                                                            objAccelerations=zero_vec)
            J_trans_pb, J_rot_pb_geom = np.array(J_trans_pb), np.array(J_rot_pb_geom)
            
            H = get_H_forEulerXYZ(orn_euler)
            J_rot_pb = np.linalg.inv(H) @ J_rot_pb_geom # analytical Jacobian from geometric Jacobian        

            # tensorflow-implementation
            _, _, J_trans_tf, _ = taskmap_joint_space_to_pos.differentiate(q=tf.constant(robot.q, dtype=tf.float32),
                                                                            qd=tf.zeros_like(robot.q, dtype=tf.float32))
            orn_euler_tf, _, J_rot_tf, _ = taskmap_joint_space_to_euler.differentiate(q=tf.constant(robot.q, dtype=tf.float32),
                                                                            qd=tf.zeros_like(robot.q, dtype=tf.float32))

            R_tf = spatial.transform.Rotation.from_euler('xyz', orn_euler_tf).as_matrix()
            assert np.max(np.abs(J_trans_pb - J_trans_tf)) < 1e-6
            assert np.max(np.abs(J_rot_pb - J_rot_tf)) < 1e-3, f'-----------FAILED WITH {np.max(np.abs(J_rot_pb - J_rot_tf))=}-----------'
            assert np.max(np.abs(R_pb - R_tf)) < 1e-4

if __name__ == '__main__':
    test_jacobians()
    print('Done!')