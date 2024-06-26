import os
import sys

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, TwoJointRobot, Goal
from rmp import RmpCore, TargetPolicy, ConfigurationSpaceBiasing
from kinematics import UrdfForwardKinematic
from taskmap import IdentityTaskmap, TaskmapByForwardKinematic, TaskmapFrom4x4ToPosition,chain_taskmaps
from helper.pybullet_helper import get_joint_order
from data_management import Datamanager
from experiments.two_joint_robot.config.camera_config import camera_view_kwargs

def run_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='02_jointspace_biasing.gif')
    simulation.connect()

    p.resetDebugVisualizerCamera(**camera_view_kwargs) # adjust camera view
    
    robot = TwoJointRobot()
    simulation.populate_scene(robot)

    goal = Goal(base_position=[1.5, 0, 0.1], radius=0.02)
    simulation.populate_scene(goal)

    # forward kinematic
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'urdf', 'TwoJointRobot_wo_fixedJoints.urdf')
    fkine = UrdfForwardKinematic(urdf_filepath=path,
                                 order=get_joint_order(robot.id, motor_joints_only=True))
    data_manager = Datamanager(fkine)

    # RMP
    core = RmpCore()

    taskmap_jointspace_to_endeffector_4x4 = TaskmapByForwardKinematic(fkine, frame='link_23')
    taskmap_endeffector_4x4_to_pos = TaskmapFrom4x4ToPosition()
    taskmap_jointspace_to_endeffector_pos = chain_taskmaps([taskmap_jointspace_to_endeffector_4x4,
                                                           taskmap_endeffector_4x4_to_pos])
    target_rmp = TargetPolicy(alpha=0.1, beta=0.5, c=0.1, goal=goal.base_position,
                              name='target', taskmap=taskmap_jointspace_to_endeffector_pos)
    core.add_rmp(target_rmp)

    jointspacebias_rmp = ConfigurationSpaceBiasing(gamma_p=0.01, gamma_d=0.1, q0=[np.pi/2, 0], name='ConfigurationSpaceBias')
    core.add_rmp(jointspacebias_rmp)

    # simulation with left bias
    for step_idx in range(10*100): # n sec at 100Hz
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            qdd_des = core.evaluate(q, qd).numpy()
        simulation.step(qdd_des)

    # simulation with right bias
    core.remove_rmp_by_name('ConfigurationSpaceBias')
    jointspacebias_rmp = ConfigurationSpaceBiasing(gamma_p=0.01, gamma_d=0.1, q0=[-np.pi/2, 0], name='ConfigurationSpaceBias')
    core.add_rmp(jointspacebias_rmp)
    robot.reset()

    for step_idx in range(10*100): # n sec at 100Hz
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            qdd_des = core.evaluate(q, qd).numpy()
        simulation.step(qdd_des)

if __name__ == '__main__':
    run_simulation()
    print('Done!')