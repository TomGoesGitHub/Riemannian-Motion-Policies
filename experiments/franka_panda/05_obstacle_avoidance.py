import os
import sys

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, Cylinder, Goal, FrankaPanda
from rmp import RmpCore#, TargetPolicy, CollisionAvoidance
from rmp2 import TargetAttractor, JointVelocityCap, JointDamping, ObstacleAvoidance
from kinematics import UrdfForwardKinematic
from taskmap import TaskmapByForwardKinematic, TaskmapFrom4x4ToPosition, TaskmapJointFrame4x4ToDistance, chain_taskmaps
from helper.pybullet_helper import get_joint_order
from experiments.franka_panda.config.camera_config import camera_view_kwargs
from experiments.franka_panda.config.scene_randomization import SceneRandomizer
from data_management import Datamanager

def work_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='05_obstacle_avoidance.gif')
    simulation.connect()
    p.resetDebugVisualizerCamera(**camera_view_kwargs) # adjust camera view


    robot = FrankaPanda()
    simulation.populate_scene(robot) #scene_randomizer.randomize_robot_config()

    goal = Goal(base_position=[0,-0.5,0.5], radius=0.02)
    simulation.populate_scene(goal)

    obstacle = Cylinder(base_position=[0.3,-0.3,0.5], base_orientation=[0.2,0,0], radius=0.025, height=0.3)
    simulation.populate_scene(obstacle)

    # forward kinematic
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'urdf', 'franka_panda', 'panda.urdf')
    fkine = UrdfForwardKinematic(urdf_filepath=path,
                                 order=get_joint_order(robot.id, motor_joints_only=True))
    data_manager = Datamanager(fkine)

    # RMP
    core = RmpCore()

    taskmap_jointspace_to_endeffector_4x4 = TaskmapByForwardKinematic(fkine, frame='panda_grasptarget_hand')
    taskmap_endeffector_4x4_to_pos = TaskmapFrom4x4ToPosition()
    taskmap_jointspace_to_endeffector_pos = chain_taskmaps([taskmap_jointspace_to_endeffector_4x4,
                                                           taskmap_endeffector_4x4_to_pos])
    # target_rmp = TargetPolicy(alpha=0.1, beta=1, c=0.1, goal=goal.base_position,
    #                           name='target', taskmap=taskmap_jointspace_to_endeffector_pos)
    target_rmp = TargetAttractor(
        goal=goal.base_position, accel_p_gain=0.1, accel_d_gain=1,
        accel_norm_eps=0.075, metric_alpha_length_scale=0.05,
        min_metric_alpha=0.03, max_metric_scalar=1, min_metric_scalar=0.5,
        proximity_metric_boost_scalar=1.,  proximity_metric_boost_length_scale=0.02,
        taskmap=taskmap_jointspace_to_endeffector_pos, name='attractor')
    core.add_rmp(target_rmp)

    joint_veclocity_cap_rmp = JointVelocityCap(
        max_velocity=0.5, velocity_damping_region=0.15, damping_gain=5.0, metric_weight=0.05
    )
    core.add_rmp(joint_veclocity_cap_rmp)

    joint_damping_rmp = JointDamping(
        accel_d_gain=1, metric_scalar=0.005, inertia=0.3
    )
    core.add_rmp(joint_damping_rmp)

    for i, frame in enumerate(fkine.frame_names):
        has_collision_shape = (p.getCollisionShapeData(robot.id, linkIndex=i) != ())
        if not has_collision_shape:
            continue
        taskmap_jointspace_to_referenceframe_4x4 = TaskmapByForwardKinematic(fkine, frame)
        taskmap_referenceframe_4x4_to_distance = TaskmapJointFrame4x4ToDistance(
            pos_on_link_in_base_frame = data_manager[frame]['pos_on_link_in_base_frame'],
            pos_on_obstacle_in_base_frame = data_manager[frame]['pos_on_obstacle_in_base_frame']
        )
        taskmap_jointspace_to_distance = chain_taskmaps([taskmap_jointspace_to_referenceframe_4x4,
                                                        taskmap_referenceframe_4x4_to_distance])
        # obstacle_rmp = CollisionAvoidance(d=data_manager[frame]['distance'], vec=data_manager[frame]['normal_vec'],
        #                                   eta_rep=0.1*np.e, nu_rep=0.3, eta_damp=0, nu_damp=0.2, r=0.8, c=1e5,
        #                                   taskmap=taskmap_jointspace_to_pos, name=f'collision_avoidance_for_{frame}')
        obstacle_rmp = ObstacleAvoidance(
            margin=0., damping_gain=50, damping_std_dev=0.04, damping_robustness_eps=0.01,
            damping_velocity_gate_length_scale=0.01, repulsion_gain=800, repulsion_std_dev=0.01,
            metric_modulation_radius=0.5, metric_scalar=1, metric_exploder_std_dev=0.02,
            metric_exploder_eps=0.001, taskmap=taskmap_jointspace_to_distance,
            name=f'collision_avoidance_for_{frame}'
        )
        core.add_rmp(obstacle_rmp)

    # simulation
    for step_idx in range(30*100): # n sec at 100Hz
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
        simulation.step(qdd_des)

if __name__ == '__main__':
    work_simulation()
    print('Done!')