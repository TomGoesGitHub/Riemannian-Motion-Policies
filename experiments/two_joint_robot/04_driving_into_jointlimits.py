import os
import sys

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, TwoJointRobot, Goal
from rmp import RmpCore, JointLimitAvoidance, TargetPolicy
from kinematics import UrdfForwardKinematic
from taskmap import IdentityTaskmap
from helper.pybullet_helper import get_joint_order
from experiments.two_joint_robot.config.camera_config import camera_view_kwargs

# class RotatePositiveRMP(RiemannianMotionPolicy):
#     def __init__(self):
#         super().__init__(name='', taskmap=IdentityTaskmap())
    
#     def _motion_command(self, x, xd):
#         q, qd = x, xd
#         qdd_des = tf.concat([[1.], tf.zeros(q.shape[-1]-1)], axis=0)
#         return qdd_des
    
#     def _metric(self, x, xd):
#         M = tf.eye(x.shape[-1])
#         return M


def run_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='04_driving_into_jointlimit.gif')
    simulation.connect()

    p.resetDebugVisualizerCamera(**camera_view_kwargs) # adjust camera view
    
    robot = TwoJointRobot()
    simulation.populate_scene(robot)
    robot.q = np.array([-np.pi/4, -np.pi/4])

    # forward kinematic
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'urdf', 'TwoJointRobot_wo_fixedJoints.urdf')
    fkine = UrdfForwardKinematic(urdf_filepath=path,
                                 order=get_joint_order(robot.id, motor_joints_only=True))

    # RMP
    core = RmpCore()
    rotate_joint1_positive_rmp = TargetPolicy(alpha=0.1, beta=1, c=0.1, goal=[robot.q_lim_low[0], 0], taskmap=IdentityTaskmap())
    core.add_rmp(rotate_joint1_positive_rmp)
    #jointlimit_rmp = JointLimitAvoidanceOld(robot.q_lim_low, robot.q_lim_high, gamma_p=0.3, gamma_d=1, lamda=0.1, c=0.1)
    jointlimit_rmp = JointLimitAvoidance(robot.q_lim_low, robot.q_lim_high, gamma_p=0.2, gamma_d=1)
    core.add_rmp(jointlimit_rmp)

    # simulation
    for step_idx in range(30*100): # n sec at 100Hz
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            qdd_des = core.evaluate(q, qd).numpy()

            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                              frame=tf.constant('link_23'))[0,:2, 3]
            x_goal = fkine.forward(tf.constant([1/2*(robot.q_lim_high+robot.q_lim_low)], dtype=tf.float32),
                              frame=tf.constant('link_23'))[0,:2, 3]
            is_solved = (np.linalg.norm(x-x_goal) < 0.02) and np.linalg.norm(robot.qd) < 0.001
            if is_solved:
                robot.q = np.random.uniform(low=robot.q_lim_low, high=robot.q_lim_high)
                robot.qd = np.zeros_like(robot.q)
        
        
        simulation.step(qdd_des)

if __name__ == '__main__':
    run_simulation()
    print('Done!')