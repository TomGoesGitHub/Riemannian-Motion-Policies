import os
import sys

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, SceneRandomizer, Goal, FrankaPanda
from rmp import RmpCore, TargetPolicy
from kinematics import UrdfForwardKinematic
from taskmap import TaskmapByForwardKinematic, TaskmapFrom4x4ToPosition, chain_taskmaps
from helper.pybullet_helper import get_joint_order
from experiments.franka_panda.config.camera_config import camera_view_kwargs
from experiments.franka_panda.config.scene_randomization import SceneRandomizer
from data_management import Datamanager

def work_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='01_target_rmp_only.gif')
    # scene_randomizer = SceneRandomizer()

    simulation.connect()
    p.resetDebugVisualizerCamera(**camera_view_kwargs) # adjust camera view
    
    robot = FrankaPanda()
    simulation.populate_scene(robot) #scene_randomizer.randomize_robot_config()

    goal = Goal(base_position=[0.6, 0, 0.4], radius=0.02)
    simulation.populate_scene(goal)

    # forward kinematic
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'urdf', 'franka_panda', 'panda.urdf')
    fkine = UrdfForwardKinematic(urdf_filepath=path,
                                 order=get_joint_order(robot.id, motor_joints_only=True))

    # RMP
    core = RmpCore()

    taskmap_jointspace_to_endeffector_4x4 = TaskmapByForwardKinematic(fkine, frame='panda_grasptarget_hand')
    taskmap_endeffector_4x4_to_pos = TaskmapFrom4x4ToPosition()
    taskmap_jointspace_to_endeffector_pos = chain_taskmaps([taskmap_jointspace_to_endeffector_4x4,
                                                           taskmap_endeffector_4x4_to_pos])
    target_rmp = TargetPolicy(alpha=0.1, beta=0.5, c=0.1, goal=goal.base_position,
                              name='target', taskmap=taskmap_jointspace_to_endeffector_pos)
    core.add_rmp(target_rmp)

    # simulation
    for step_idx in range(30*100): # n sec at 100Hz
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            qdd_des = core.evaluate(q, qd).numpy()

            # reposition goal if solved
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
            if is_solved:
                goal.base_position = np.random.uniform(low=[0.3,-0.7, 0.3], high=[0.7, 0.7, 0.7]).tolist()
                target_rmp.goal = np.array(goal.base_position)
        simulation.step(qdd_des)



    print('Done!')

if __name__ == '__main__':
    work_simulation()