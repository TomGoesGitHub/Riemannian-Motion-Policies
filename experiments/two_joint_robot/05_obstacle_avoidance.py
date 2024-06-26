import os
import sys

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, TwoJointRobot, Goal, Cylinder
from rmp import RmpCore, TargetPolicy, CollisionAvoidance, ConfigurationSpaceBiasing
from kinematics import UrdfForwardKinematic
from taskmap import TaskmapByForwardKinematic, TaskmapRelative4x4, TaskmapFrom4x4ToPosition, IdentityTaskmap, chain_taskmaps
from helper.pybullet_helper import get_joint_order
from helper.tensorflow_helper import as_batch
from data_management import Datamanager
from experiments.two_joint_robot.config.camera_config import camera_view_kwargs

def run_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='05_obstacle_avoidance.gif')
    simulation.connect()

    p.resetDebugVisualizerCamera(**camera_view_kwargs) # adjust camera view
    
    robot = TwoJointRobot()
    simulation.populate_scene(robot)

    goal = Goal(base_position=[1.4, -1.4, 0.1], radius=0.02)
    simulation.populate_scene(goal)

    obstacle = Cylinder(base_position=[1.6, -0.8, 0], base_orientation=[0,0,0,1], height=0.8, radius=0.1)
    simulation.populate_scene(obstacle)

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
    target_rmp = TargetPolicy(alpha=0.1, beta=0.1, c=0.1, goal=goal.base_position,
                              name='target', taskmap=taskmap_jointspace_to_endeffector_pos)
    core.add_rmp(target_rmp)

    for frame in fkine.frame_names:
        taskmap_jointspace_to_referenceframe_4x4 = TaskmapByForwardKinematic(fkine, frame)
        taskmap_referenceframe_to_targetframe_4x4 = TaskmapRelative4x4(relative_pos=data_manager[frame]['relative_position'])
        taskmap_targetframe_4x4_to_pos = TaskmapFrom4x4ToPosition()
        taskmap_jointspace_to_pos = chain_taskmaps([taskmap_jointspace_to_referenceframe_4x4,
                                                    taskmap_referenceframe_to_targetframe_4x4,
                                                    taskmap_targetframe_4x4_to_pos])
        obstacle_rmp = CollisionAvoidance(d=data_manager[frame]['distance'], vec=data_manager[frame]['normal_vec'],
                                          eta_rep=0.1*np.e, nu_rep=0.3, eta_damp=1, nu_damp=0.3, r=1.1, c=1e5,
                                          taskmap=taskmap_jointspace_to_pos, name=f'collision_avoidance_for_{frame}')
        core.add_rmp(obstacle_rmp)

    print(core)

    # simulation
    for step_idx in range(100*100): # n sec at 100Hz
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()

            # reposition goal if solved
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32), frame=tf.constant('link_23'))[0, :2, 3]
            x_goal = np.array(goal.base_position)[:2]
            is_solved = np.linalg.norm(x-x_goal) < 0.02
            # if is_solved:
            #     new_goal_pos = np.random.uniform(low=[0.1, 0.1, 0.1], high=[1.4, 1.4, 0.1])
            #     goal.base_position = new_goal_pos.tolist()
            #     target_rmp.goal = np.array(goal.base_position)
        simulation.step(qdd_des)

if __name__ == '__main__':
    run_simulation()
    print('Done!')