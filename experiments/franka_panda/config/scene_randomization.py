import random
import os
import sys

import numpy as np

sys.path.append(os.path.join(*3*[os.pardir]))
from simulation import FrankaPanda, Cylinder


class SceneRandomizer:
    default_sample_space = [
        (Cylinder, {'base_position_cylindrical' : {'low': [0.4, 0, 0], 'high': [0.9, 2*np.pi, 1]},
                    'base_orientation': {'low': np.zeros(shape=3), 'high': np.full(fill_value=np.pi, shape=3)},
                    'radius': {'low': 0.05, 'high': 0.1},
                    'height': {'low': 0.5, 'high': 0.5}}),
    ]

    default_robot_sample_space = (
        FrankaPanda, {'q' : {'low': FrankaPanda.q_ready - 0.1, 'high': FrankaPanda.q_ready + 0.1},
                      'qd' : {'low': np.full_like(FrankaPanda.q_ready, fill_value=-0.005),
                              'high': np.full_like(FrankaPanda.q_ready, fill_value=0.005)}}
    )

    default_goal_sample_space = {'base_position_cylindrical' : {'low': [0.4, 0, 0], 'high': [0.9, 2*np.pi, 1]}}

    def __init__(self, sample_space=None):
        if sample_space is None:
            sample_space = self.default_sample_space
        self.sample_space = sample_space
        

    def randomize_obstacles(self, n_obstacles):
        obstacles = []
        for _ in range(n_obstacles):
            obstacle_class, param_space = random.sample(population=self.default_sample_space, k=1)[0]

            is_cartesian = 'base_position' in param_space.keys()
            is_cylindrical = 'base_position_cylindrical' in param_space.keys()
            assert is_cartesian ^ is_cylindrical

            params = {key : np.random.uniform(**param_space[key])
                      for key in param_space.keys()}
            
            if is_cylindrical:
                r, phi, z = params['base_position_cylindrical']
                params['base_position'] = [r*np.cos(phi), r*np.sin(phi), z]
                params.pop('base_position_cylindrical')

            obstacle = obstacle_class(**params)
            obstacles.append(obstacle)
        return obstacles
    
    def randomize_robot_config(self):
        robot_class, param_space = self.default_robot_sample_space
        params = {key : np.random.uniform(**param_space[key]) for key in param_space.keys()}
        robot = robot_class(**params)
        return robot
    
    def randomize_goal(self, goal):
        param_space = self.default_goal_sample_space
        params = {key : np.random.uniform(**param_space[key]) for key in param_space.keys()}
        r, phi, z = params['base_position_cylindrical']
        x, y = r*np.cos(phi), r*np.sin(phi)
        goal.base_position = [x,y,z]
