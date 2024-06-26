import os
import sys

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, Cylinder, Goal, FrankaPanda
from rmp import RmpCore#, TargetPolicy, CollisionAvoidance
from rmp2 import TargetAttractor, JointVelocityCap, JointDamping, ObstacleAvoidance, CSpaceBiasing
from kinematics import UrdfForwardKinematic
from taskmap import TaskmapByForwardKinematic, TaskmapFrom4x4ToPosition, TaskmapJointFrame4x4ToDistance, chain_taskmaps
from helper.pybullet_helper import get_joint_order
from experiments.franka_panda.config.camera_config import camera_view_kwargs
from data_management import Datamanager

def step_camera():
    camera_info = p.getDebugVisualizerCamera()
    yaw, pitch, dist, target = camera_info[8:]
    yaw += 360/3300 # step
    yaw = np.mod(yaw, 360)
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)


def work_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='06_cluttered_environment.gif')
    simulation.connect()
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-35.0, cameraTargetPosition=[0,0,0]) # adjust camera view


    robot = FrankaPanda()
    simulation.populate_scene(robot) #scene_randomizer.randomize_robot_config()

    goal = Goal(base_position=[0.2,-0.2,0.5], radius=0.02)
    simulation.populate_scene(goal)

    obstacles = [
        Cylinder(base_position=[0.35,-0.2, 0.55], base_orientation=[0.1,0,0], radius=0.025, height=0.2),
        Cylinder(base_position=[0.1,-0.4, 0.125], base_orientation=[0.1,0,0], radius=0.025, height=0.3),
        Cylinder(base_position=[0.33,-0.3, 0.7], base_orientation=[-1.7,0.7,0], radius=0.025, height=0.3),

        Cylinder(base_position=[0.55, 0.5-0.25, 0.5], base_orientation=[0.1,0,0], radius=0.025, height=0.3),
        Cylinder(base_position=[0.8, 0.5-0.25, 0.3], base_orientation=[0.1,0,0], radius=0.025, height=0.3),

        # Cylinder(base_position=[0.55, 0.5-0.25, 0.5], base_orientation=[0.1,0,0], radius=0.025, height=0.3),
        # Cylinder(base_position=[0.8, 0.5-0.25, 0.2], base_orientation=[0.1,0,0], radius=0.025, height=0.3),
        
        Cylinder(base_position=[0.5,  0.5-0.1, 0.31], base_orientation=[3.14/2,0,0], radius=0.025, height=0.3),
        Cylinder(base_position=[0.35+0.1,  0.5-0.4, 0.11], base_orientation=[3.14/2,0,0], radius=0.025, height=0.3),
    ]
    simulation.populate_scene(obstacles)

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
        goal=goal.base_position, accel_p_gain=0.3, accel_d_gain=0.6,
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

    cspace_bias_rmp = CSpaceBiasing(
        goal=tf.constant([0.0, -0.9, 0.0, -2.8, 0.0, 2.0, 0.7853981633974483, 0.02, 0.02]),
        metric_scalar=0.005, position_gain=1, damping_gain=2,
        robust_position_term_thresh=0.5, inertia=0.0001,
    )
    core.add_rmp(cspace_bias_rmp)

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
    step_idx, is_solved = 0, False
    while not is_solved:
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
        simulation.step(qdd_des)
        step_camera()
        step_idx += 1
    print(robot.q)
    print(robot.qd)
    
    # # robot.q = [-2.55193172e-01, -1.03149144e+00, -8.25478290e-02, -2.86239616e+00,
    # #             -1.53745719e+00,  1.66907480e+00,  1.11192508e+00,  3.98454676e-02,
    # #             -1.02094421e-23]
    # # robot.qd =[ 2.14332394e-03, -6.64037461e-02, -9.86218731e-02, -1.11694799e-02,
    # #             -8.24992276e-02,  1.29430754e-01, -7.50460772e-03,  4.43362615e-03,
    # #             -8.47032947e-22]
    goal.base_position, is_solved = [0.5,-0.4,0.5], False
    target_rmp.goal = goal.base_position
    while not is_solved:
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
        simulation.step(qdd_des)
        step_camera()
        step_idx += 1
    print(robot.q)
    print(robot.qd)


    # # robot.q = [ 8.39781468e-01 -9.33406557e-01 -1.42816076e+00 -2.05601378e+00
    # # -2.01557260e+00  3.00819648e+00  9.14700780e-01  2.79012499e-03
    # # 6.69453406e-05]
    # # robot.qd = [ 0.13035623 -0.00514884 -0.06725104  0.10215527 -0.00578815  0.0782796
    # #             -0.01065427 -0.00421501  0.00047359]
    goal.base_position, is_solved = [0.6, -0.2, 0.7], False
    target_rmp.goal = goal.base_position
    while not is_solved:
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
        simulation.step(qdd_des)
        step_camera()
        step_idx += 1
    print(robot.q)
    print(robot.qd)


    # robot.q = [ 1.79176118e+00, -1.15584907e+00, -1.49914638e+00, -1.17060426e+00,
    #             -2.30321492e+00,  3.82230004e+00,  6.18781794e-01, -2.52772611e-22,
    #             -1.86871299e-07] 
    # robot.qd = [ 1.47789348e-02, -2.17616828e-02,  2.74375612e-02,  1.71488040e-02,
    #             -3.40455525e-02,  8.10604864e-09, -1.84845877e-02, -6.77626358e-21,
    #             1.13045345e-07]

    goal.base_position, is_solved = [0.6, 0, 0.3], False
    target_rmp.goal = goal.base_position
    while not is_solved:
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
        simulation.step(qdd_des)
        step_camera()
        step_idx += 1
    print(robot.q)
    print(robot.qd)


    goal.base_position, is_solved = [0.4, 0.55, 0.65], False
    target_rmp.goal = goal.base_position
    while not is_solved:
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
        simulation.step(qdd_des)
        step_camera()
        step_idx += 1
    print(robot.q)
    print(robot.qd)

    goal.base_position, is_solved = [0.65, 0.35, 0.65], False
    target_rmp.goal = goal.base_position
    while not is_solved:
        if (step_idx % 10 == 0): # rmp-control at 10 Hz
            q, qd, distance_data = simulation.state()
            data_manager.update(q, distance_data)
            qdd_des = core.evaluate(q, qd).numpy()
            x = fkine.forward(tf.constant([robot.q], dtype=tf.float32),
                            frame=tf.constant('panda_grasptarget_hand'))[0,:3, 3]
            x_goal = np.array(goal.base_position)
            is_solved = np.linalg.norm(x-x_goal) < 0.02
        simulation.step(qdd_des)
        step_camera()
        step_idx += 1
    print(robot.q)
    print(robot.qd)
    print('\n', step_idx)


if __name__ == '__main__':
    work_simulation()
    print('Done!')