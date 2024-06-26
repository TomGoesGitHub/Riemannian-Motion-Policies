import os
import sys
import cProfile # todo: tmp

import numpy as np
import tensorflow as tf
import pybullet as p

sys.path.append(os.path.join(*2*[os.pardir]))
from simulation import Simulation, SceneRandomizer, Goal
from rmp import RmpCore, TargetPolicy, CollisionAvoidance
from kinematics import UrdfForwardKinematic
from helper.pybullet_helper import get_joint_order
from experiments.franka_panda.config.camera_config import camera_view_kwargs
from experiments.franka_panda.config.scene_randomization import SceneRandomizer

def work_simulation():
    # environment
    simulation = Simulation(delta_t=0.01, animation_save_path='03_self_avoidance.gif')
    simulation.connect()
    p.resetDebugVisualizerCamera(**camera_view_kwargs) # adjust camera view
    
    scene_randomizer = SceneRandomizer()
    robot = scene_randomizer.randomize_robot_config()
    simulation.populate_scene(robot)

    goal = Goal(base_position=[-0.5, 0, 0.25], radius=0.02)
    simulation.populate_scene(goal)
    
    # rmp
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'urdf', 'franka_panda', 'panda.urdf')
    fkine = UrdfForwardKinematic(urdf_filepath=path,
                                 order=get_joint_order(robot.id, motor_joints_only=True))
    core = RmpCore(fkine)

    target_rmp = TargetPolicy(alpha=1, beta=10, c=0.1, goal=goal.base_position, name='target', reference_frame='panda_grasptarget_hand')
    collsion_avoidance_rmp = CollisionAvoidance(r=0.2, eta_rep=1, nu_rep=0.1, eta_damp=1, nu_damp=0.1, beta=0, c=0.1)
    core.add_rmp(collsion_avoidance_rmp)
    core.add_rmp(target_rmp)

    # simulation
    for _ in range(10*100): # n sec at 100Hz
        state = simulation.state()
        core.update(*state)
        qdd_des = core.evaluate().numpy()
        if _ == 1:
            with cProfile.Profile() as pr:
                core.evaluate().numpy()
            pr.dump_stats('tmp.prof')
            print('.....')
        simulation.step(qdd_des)

    print('Done!')

if __name__ == '__main__':
    work_simulation()