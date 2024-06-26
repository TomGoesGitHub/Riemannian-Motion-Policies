import os
from abc import ABC, abstractmethod
import random
import imageio

import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation

from helper.pybullet_helper import get_motor_joints, get_joint_order, get_kinematic_chains, check_link_neighborhood


class PyBulletObject(ABC):
    def __init__(self, base_position, base_orientation, fixed_base: bool = True):
        # Store initial position and orientation for resets
        self.init_base_position = base_position
        self.init_base_orientation = base_orientation
        self.fixed_base = fixed_base

        # Stick to base_position/_orientation for conformity with Robot definitions
        self._base_position = base_position
        self._base_orientation = base_orientation

        self.id = None  # To be filled by Bullet client

    @property
    def quaternions(self):
        if len(self._base_orientation) == 3:
            return p.getQuaternionFromEuler(self._base_orientation)
        else:
            return self._base_orientation

    @property
    def euler(self):  # Fixed XYZ (RPY) rotation order
        if len(self._base_orientation) == 3:
            return self._base_orientation
        else:
            return p.getEulerFromQuaternion(self._base_orientation)

    @property
    def base_orientation(self):
        return self.quaternions

    @base_orientation.setter
    def base_orientation(self, values):
        assert len(values) in [3,4]
        
        if len(values) == 3:
            self._base_orientation = p.getQuaternionFromEuler(values)       
        else:
            self._base_orientation = values
        
        if hasattr(self, 'client_id'):
            self.client_id.resetBasePositionAndOrientation(self.id, self.base_position, self.base_orientation)

    @property
    def base_position(self):
        return self._base_position

    @base_position.setter
    def base_position(self, values):
        self._base_position = values
        if hasattr(self, 'client_id'):
            p.resetBasePositionAndOrientation(self.id, self._base_position, self.base_orientation, physicsClientId=self.client_id)
            #self.client_id.resetBasePositionAndOrientation(self.id, self._base_position, self.base_orientation)

    def reset(self):
        self._base_orientation = self.init_base_orientation
        self.base_position = self.init_base_position
        if hasattr(self, 'client_id'):
            self.load2client(self.client_id)
        else:
            self.id = None

    @abstractmethod
    def load2client(self, client_id):
        pass # todo: check if class is abstract

    def remove_from_client(self):
        pass # todo

class TwoJointRobot(PyBulletObject):
    q_ready = np.array([0, 0]) # ready pose
    q_lim_low = np.array([-np.pi, -np.pi]) # lower joint limit
    q_lim_high = np.array([+np.pi, +np.pi]) # upper joint limit

    def __init__(self, base_position=[0,0,0], base_orientation=[0,0,0,1], q = None, qd=None):
        super(TwoJointRobot, self).__init__(base_position, base_orientation, fixed_base=True)

        # joint state
        self.init_q = q if q is not None else self.q_ready
        self.init_qd = qd if qd is not None else np.zeros_like(self.init_q)
    
    @property
    def q(self):
        q = [p.getJointState(self.id, i)[0] for i in self.idx_controllable]
        return np.array(q)
        
    @q.setter
    def q(self, q):
        for i, q_i in zip(self.idx_controllable, q):
            p.resetJointState(self.id, i, q_i)

    @property
    def qd(self):
        qd = [p.getJointState(self.id, i)[1] for i in self.idx_controllable]
        return np.array(qd)
    
    @qd.setter
    def qd(self, qd):
        for i, q_i, qd_i in zip(self.idx_controllable, self.q, qd):
            p.resetJointState(self.id, i, q_i, qd_i)

    def reset(self, q=None, qd=None):
        if q is not None:
             assert q.shape == self.q_ready.shape
             self.q = q 
        else: 
            self.q = self.q_ready

        if qd is not None:
             assert qd.shape == self.q_ready.shape
             self.qd = qd
        else: 
            self.qd = np.zeros_like(self.q_ready)
    
    def load2client(self, client_id):
        # load the robot into pybullet
        path = os.path.join(os.path.dirname(__file__), 'urdf', 'TwoJointRobot_wo_fixedJoints.urdf')
        self.id = p.loadURDF(fileName=path, useFixedBase=True, physicsClientId=client_id)
        self.mask_controllable = get_motor_joints(id=self.id)
        self.idx_controllable = [i for i, m in enumerate(self.mask_controllable) if m]
        self.reset()

class FrankaPanda(PyBulletObject):
    q_ready = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.02, 0.02]) # ready pose
    q_lim_low = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.0, 0.0, 0.0, 0.0, 0.0]) # lower joint limit
    q_lim_high = np.array([2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0, 0.0, 0.04, 0.04, 0.0]) # upper joint limit
    

    def __init__(self, base_position=[0,0,0], base_orientation=[0,0,0,1], q = None, qd=None):
        super(FrankaPanda, self).__init__(base_position, base_orientation, fixed_base=True)

        # joint state
        self.init_q = q if q is not None else self.q_ready
        self.init_qd = qd if qd is not None else np.zeros_like(self.init_q)
    
    @property
    def q(self):
        q = [p.getJointState(self.id, i)[0] for i in self.idx_controllable]
        return np.array(q)
        
    @q.setter
    def q(self, q):
        for i, q_i in zip(self.idx_controllable, q):
            p.resetJointState(self.id, i, q_i)
            # todo: check if qd is changed to a default value or kept at its current value
    
    @property
    def qd(self):
        qd = [p.getJointState(self.id, i)[1] for i in self.idx_controllable]
        return np.array(qd)
    
    @qd.setter
    def qd(self, qd):
        for i, q_i, qd_i in zip(self.idx_controllable, self.q, qd):
            p.resetJointState(self.id, i, q_i, qd_i)
    
    def reset(self, q=None, qd=None):
        if q is not None:
             assert q.shape == self.q_ready.shape
             self.q = q 
        else: 
            self.q = self.q_ready

        if qd is not None:
             assert qd.shape == self.q_ready.shape
             self.qd = qd
        else: 
            self.qd = np.zeros_like(self.q_ready)
    
    def load2client(self, client_id):
        # load the robot into pybullet
        path = os.path.join(os.path.dirname(__file__), 'urdf', 'franka_panda', 'panda.urdf')
        self.id = p.loadURDF(fileName=path, useFixedBase=True, physicsClientId=client_id)
        self.mask_controllable = get_motor_joints(id=self.id)
        self.idx_controllable = [i for i, m in enumerate(self.mask_controllable) if m]
        self.reset()
        self._print_link_info()

    def _print_link_info(self):
        print('\n'+'Pybullet joint order:')
        for i in range(p.getNumJoints(self.id)):
            joint_name = p.getJointInfo(self.id, jointIndex=i)[1]
            print(i, joint_name)
        print('\n')

class Sphere(PyBulletObject):
    rgba_colors = [0.2, 0.2, 0.2, 1]

    def __init__(self, base_position, radius):
        super(Sphere, self).__init__(base_position, [0, 0, 0, 1], True)

        self.radius = radius

    @property
    def collision_shape_id(self):
        return p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)

    @property
    def visual_shape_id(self):
        return p.createVisualShape(p.GEOM_SPHERE,
                                   radius=self.radius,
                                   rgbaColor=self.rgba_colors,
                                   specularColor=[0.4, .4, 0])

    def load2client(self, client_id):
        self.id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=self.collision_shape_id,
                                    baseVisualShapeIndex=self.visual_shape_id,
                                    basePosition=self.base_position,
                                    baseOrientation=self.base_orientation,
                                    physicsClientId=client_id)
        setattr(self, 'client_id', client_id)

class Goal(Sphere):
    rgba_colors = [0, 0, 1, 1]

    @property
    def collision_shape_id(self):
        return -1 # no collision shape

    # def load2client(self, client_id):
    #     collision_shape_id = self.collision_shape_id
    #     self.id = p.createMultiBody(baseMass=0,
    #                                 baseCollisionShapeIndex=-1,
    #                                 baseVisualShapeIndex=self.visual_shape_id,
    #                                 basePosition=self.base_position,
    #                                 baseOrientation=self.base_orientation,
    #                                 physicsClientId=client_id)
    #     #p.removeCollisionShape(collision_shape_id, physicsClientId=client_id)
    #     setattr(self, 'client_id', client_id)

class Cylinder(PyBulletObject):
    def __init__(self, base_position, base_orientation, radius, height):
        super(Cylinder, self).__init__(base_position, base_orientation, True)

        self.radius = radius
        self.height = height

    @property
    def collision_shape_id(self):
        return p.createCollisionShape(p.GEOM_CYLINDER, radius=self.radius, height=self.height)

    @property
    def visual_shape_id(self):
        return p.createVisualShape(p.GEOM_CYLINDER,
                                   radius=self.radius,
                                   length=self.height,
                                   rgbaColor=[0.2, 0.2, 0.2, 1],
                                   specularColor=[0.4, .4, 0])

    def load2client(self, client_id):
        self.id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=self.collision_shape_id,
                                    baseVisualShapeIndex=self.visual_shape_id,
                                    basePosition=self.base_position,
                                    baseOrientation=self.base_orientation,
                                    physicsClientId=client_id)
        
        setattr(self, 'client_id', client_id)

class Simulation:
    def __init__(self, delta_t=0.01, animation_save_path=None):        
        # time 
        self._delta_t = delta_t
        self.t = 0
        
        # objects
        self.obstacles = []
        self.robot = None

        # distance-calculation
        self.todo_distances_to_links = None
        self.todo_distances_to_obstacles = None

        # animation
        self.animation_save_path = animation_save_path
        self._is_animated = (self.animation_save_path is not None)
        self._fps_animation = 16
        self._t_prev_animation = 0
        if self._is_animated:
            self._writer = imageio.get_writer(self.animation_save_path, mode='I', fps=self._fps_animation, loop=0)
        
    def _capture_frame(self):
        camera_settings = p.getDebugVisualizerCamera()
        width, height, viewMatrix, projectionMatrix = camera_settings[:4]
        frame = p.getCameraImage(width, height, viewMatrix, projectionMatrix, renderer=p.ER_TINY_RENDERER)[2]
        self._writer.append_data(frame) 

    def _reset_distance_calculation(func):
        def wrapper(self, *args, **kwargs):
            self.todo_distances_to_links = None
            self.todo_distances_to_obstacles = None
            result = func(self, *args, **kwargs)
            return result
        return wrapper
    
    @property
    def n_obstacles(self):
        return len(self.obstacles)

    @property
    def delta_t(self):
        return self._delta_t
    
    @delta_t.setter
    def delta_t(self, value):
        self._delta_t = value
        if self.client_id:
            p.setTimeStep(self.delta_t)
    
    @_reset_distance_calculation
    def connect(self):
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.setTimeStep(self.delta_t)
        self.plane_id = p.loadURDF('plane.urdf')
    
    @_reset_distance_calculation
    def populate_scene(self, objects):
        if type(objects) is not list:
            objects = [objects]
        
        for object in objects:
            object.load2client(client_id=self.client_id)
            if isinstance(object, Goal):
                continue
            elif isinstance(object, FrankaPanda) or isinstance(object, TwoJointRobot): # todo: generalize to Robot class
                self.robot = object
                p.setJointMotorControlArray(self.robot.id,
                                            self.robot.idx_controllable,
                                            p.VELOCITY_CONTROL,
                                            forces=np.zeros_like(self.robot.idx_controllable))
            else:
                self.obstacles.append(object)

    @_reset_distance_calculation
    def clear_scene(self):
        if self.client_id:
            for obstacle in self.obstacles:
                p.removeBody(bodyUniqueId=obstacle.id, physicsClientId=self.client_id)
            self.obstacles = []

            p.removeBody(bodyUniqueId=self.robot.id, physicsClientId=self.client_id)
            self.robot = None
        else:
            print('Not connected yet...')
    
    @_reset_distance_calculation
    def disconect(self):
        self.clear_scene()
        self.reset()
        p.disconect(physicsClientId=self.client_id)
        self.client_id = None

    def step(self, qdd_desired=None):
        forces = p.calculateInverseDynamics(bodyUniqueId=self.robot.id,
                                            objPositions=self.robot.q.tolist(),
                                            objVelocities=self.robot.qd.tolist(),
                                            objAccelerations=qdd_desired.tolist(),
                                            physicsClientId=self.client_id)
        
        p.setJointMotorControlArray(bodyUniqueId=self.robot.id,
                                    jointIndices=self.robot.idx_controllable,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=forces,
                                    physicsClientId=self.client_id)
        p.stepSimulation(self.client_id)
        self.t += self.delta_t
        
        if self._is_animated and (self.t > self._t_prev_animation + 1/self._fps_animation):
            self._capture_frame()
            self._t_prev_animation = self.t
            
    def state(self):
        q, qd = self._get_joint_state()
        distances = self._get_distances_state()
        return q.astype(np.float32), qd.astype(np.float32), distances

    def _get_joint_state(self):
        q = self.robot.q
        qd = self.robot.qd
        return q, qd

    def _get_distances_state(self):
        # update todo lists for distance calculation
        if self.todo_distances_to_links is None:
            self._update_todolist_for_distances_to_links()
        if self.todo_distances_to_obstacles is None:
            self._update_todolist_for_distances_to_obstacles()
        
        # calculate distances
        # result = self.calculate_distances(calculations=self.todo_distances_to_links)
        # result.extend(self.calculate_distances(calculations=self.todo_distances_to_obstacles)) # todo: self-avoidance does not work properly yet
        result = self.calculate_distances(calculations=self.todo_distances_to_obstacles)
        return result

    def _update_todolist_for_distances_to_links(self):
        todo_list = []
        n_joints =  p.getNumJoints(self.robot.id) # motor joints and fixed joints
        for link_index_A in range(n_joints): # the base (link_index=-1) is assumed to be fixed
                                             # and therefore we dont need a RMP for it
            for link_index_B in range(-1, n_joints): # todo: maybe range(-1, link_index_A)            
                # only do distance calculations if both links have a collision shape
                collision_info_A = p.getCollisionShapeData(objectUniqueId=self.robot.id, linkIndex=link_index_A)
                collision_info_B = p.getCollisionShapeData(objectUniqueId=self.robot.id, linkIndex=link_index_B)
                has_collsion_A = (collision_info_A != ())
                has_collsion_B = (collision_info_B != ())
                if not (has_collsion_A and has_collsion_B):
                    continue
                    
                # only consider pairs, which have a certain distance in the kinematic chain
                shared_neighborhood1 = check_link_neighborhood(self.robot.id, link_index_A, link_index_B, n_neighbors=3)
                shared_neighborhood2 = check_link_neighborhood(self.robot.id, link_index_B, link_index_A, n_neighbors=3)
                shared_neighborhood = (shared_neighborhood1 or shared_neighborhood2)
                if shared_neighborhood:
                    continue
                
                # valid tuple
                robot_id, robot_link_index = self.robot.id, link_index_A
                obstacle_id, obstacle_link_index = self.robot.id, link_index_B # linkB is interpretated as obstacle
                joint_name = p.getJointInfo(self.robot.id, robot_link_index)[1]
                other_joint_name = p.getJointInfo(self.robot.id, link_index_B)[1] if link_index_B != -1 else 'base'
                description = f'{joint_name} to {other_joint_name}'
                tuple = (joint_name, robot_id, robot_link_index, obstacle_id, obstacle_link_index, description)
                #(robot_link_index, obstacle_link_index) # todo: remove print
                todo_list.append(tuple)
        self.todo_distances_to_links = todo_list
    
    def _update_todolist_for_distances_to_obstacles(self):
        todo_list = []
        n_joints = p.getNumJoints(self.robot.id) # motor joints and fixed joints
        for robot_link_index in range(n_joints):
            has_collision = p.getCollisionShapeData(objectUniqueId=self.robot.id, linkIndex=robot_link_index)
            if not has_collision:
                continue
            for obstacle in self.obstacles:
                obstacle_has_collision = p.getCollisionShapeData(objectUniqueId=obstacle.id, linkIndex=-1)
                if not obstacle_has_collision:
                    continue
                joint_name = p.getJointInfo(self.robot.id, robot_link_index)[1]
                obstacle_link_index = -1 # the obstacle is not a robot, therefore
                                         # it consists of base (linkIndex=-1) only
                description = f'{joint_name} to obstacle with id={obstacle.id}'
                tuple = (joint_name, self.robot.id, robot_link_index, obstacle.id, obstacle_link_index, description)
                todo_list.append(tuple)
        self.todo_distances_to_obstacles = todo_list

    def calculate_distances(self, calculations):
        result = []
        for tuple in calculations:
            joint_name, robot_id, robot_link_index, obstacle_id, obstacle_link_index, description = tuple # unpack
            joint_name = joint_name.decode('ascii')
            info = p.getClosestPoints(bodyA=robot_id,
                                      bodyB=obstacle_id,
                                      distance=np.inf,
                                      linkIndexA=robot_link_index,
                                      linkIndexB=obstacle_link_index,)
            pos_on_link_in_world, pos_on_obstacle_in_world, normal_vec_in_world, distance = info[0][5:9]

            # spatial transformation:
            R_world_base = np.array(p.getMatrixFromQuaternion(self.robot.base_orientation)).reshape(3,3)
            pos_of_base_in_world = np.array(self.robot.base_position).reshape(3,) 
            R_base_world = R_world_base.T
            pos_on_link_in_base_frame = R_base_world @ (pos_on_link_in_world - pos_of_base_in_world)
            normal_vec_in_base_frame = R_base_world @ normal_vec_in_world

            result.append((joint_name, pos_on_link_in_base_frame.astype(np.float32),
                           np.array(pos_on_obstacle_in_world, dtype=np.float32),
                           normal_vec_in_base_frame.astype(np.float32), distance, description))
        return result

    def reset(self):
        self.t = 0
        for obstacle in self.obstacles:
            obstacle.reset()
        if self.robot:
            self.robot.reset()


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
