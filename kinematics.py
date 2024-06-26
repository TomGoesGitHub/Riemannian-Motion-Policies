
import tensorflow as tf
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from helper.urdf_parsing import UrdfTree
from helper.pybullet_helper import get_kinematic_chains
from helper.trigonometry_helper import get_H_forEulerXYZ #, get_H_fromQuaternions
from helper.rmp_helper import jacobian_vector_product

@tf.function(input_signature=[tf.TensorSpec(shape=[ None, 4, 4], dtype=tf.float32)])
def reduce_matrix_prod(all_T):
    i0 = tf.constant(0)
    batch_shape = [tf.shape(all_T)[0]]
    m0 = tf.eye(4, batch_shape=batch_shape[:-1], dtype=tf.float32) # note: the final batch dimension will be reduced
    i, m = i0, m0
    cond = lambda i, _: tf.less(i, batch_shape[-1])
    body = lambda i, m: (i + 1, m @ all_T[..., i, :,:])
    return tf.while_loop(cond, body, loop_vars=[i, m], shape_invariants=[i0.shape, m0.shape])[1]

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
def R_x(angle):
    batch_size = tf.shape(angle)[0]
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    z = tf.zeros_like(c)
    top_row = tf.broadcast_to([[1.,0.,0.]], shape=[batch_size, 1, 3])
    middle_row = tf.stack([z, c, -s], axis=-1)
    bottom_row = tf.stack([z, s, c], axis=-1)
    R = tf.concat([top_row, middle_row, bottom_row], axis=-2)
    return R

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
def R_y(angle):
    batch_size = tf.shape(angle)[0]
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    z = tf.zeros_like(c)
    top_row = tf.stack([c, z, s], axis=-1)
    middle_row = tf.broadcast_to([[0.,1.,0.]], shape=[batch_size, 1, 3])
    bottom_row = tf.stack([-s, z, c], axis=-1)
    R = tf.concat([top_row, middle_row, bottom_row], axis=-2)
    return R

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
def R_z(angle):
    batch_size = tf.shape(angle)[0]
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    z = tf.zeros_like(c)
    top_row = tf.stack([c, -s, z], axis=-1)
    middle_row = tf.stack([s, c, z], axis=-1)
    bottom_row = tf.broadcast_to([[0.,0.,1.]], shape=[batch_size, 1, 3])
    R = tf.concat([top_row, middle_row, bottom_row], axis=-2)
    return R

@tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 3,3], dtype=tf.float32),
                         tf.TensorSpec(shape=[None,3], dtype=tf.float32)]
)
def homogenous_transformation(R, t):
    tf.assert_equal(R.shape[-2:], [3,3])
    tf.assert_equal(t.shape[-1], 3)
    
    batch_size = tf.shape(R)[0]
    t = t[..., tf.newaxis]
    Rt = tf.concat([R, t], axis=-1)
    bottom = tf.concat([tf.zeros([batch_size, 1, 3]), tf.ones([batch_size, 1, 1])], axis=-1)
    T = tf.concat([Rt, bottom], axis=-2)
    return T


def euler_from_rotation_matrix(rotation_matrix):
    '''ChatGPT'''
    rotation_matrix = tf.cast(rotation_matrix, dtype=tf.float32)
    
    r00 = rotation_matrix[:, 0, 0]
    r10 = rotation_matrix[:, 1, 0]
    r21 = rotation_matrix[:, 2, 1]
    r22 = rotation_matrix[:, 2, 2]
    r20 = rotation_matrix[:, 2, 0]

    theta_y = -tf.asin(r20)
    cos_theta_y = tf.cos(theta_y)
    
    # Prevent division by zero in case of gimbal lock
    safe_cos_theta_y = tf.where(tf.abs(cos_theta_y) < 1e-6, tf.ones_like(cos_theta_y), cos_theta_y)
    # safe_cos_theta_y = tf.where(tf.abs(cos_theta_y) < 1e-6, cos_theta_y + 1e-6, cos_theta_y)
    
    theta_z = tf.atan2(r10 / safe_cos_theta_y, r00 / safe_cos_theta_y)
    theta_x = tf.atan2(r21 / safe_cos_theta_y, r22 / safe_cos_theta_y)
    
    angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
    #angles = tf.math.mod(angles, tf.constant(2*np.pi, dtype=tf.float32))
    return angles


@tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                         tf.TensorSpec(shape=[None], dtype=tf.float32)]
)
def rotation_matrix_from_rotation_vector(vec, angle):
    batch_size = tf.shape(vec)[0]
    tf.assert_equal(tf.shape(angle)[0], batch_size)

    cos = tf.math.cos(angle)[..., tf.newaxis, tf.newaxis]
    sin = tf.math.sin(angle)[..., tf.newaxis, tf.newaxis]

    zeros = tf.zeros(shape=(batch_size, 1))
    vec_with_zeros = tf.concat([zeros, vec], axis=-1)

    eye = tf.eye(3, batch_shape=[batch_size])
    outer = tf.einsum('...i,...j->...ij', vec, vec)
    
    sign = tf.constant([[1, -1, 1], [1, 1,-1], [-1, 1, 1]], dtype=tf.float32)
    where = tf.constant([[0, 3, 2], [3, 0, 1], [2, 1, 0]], dtype=tf.int32)
    u_tilde = sign * tf.gather(vec_with_zeros, where, axis=-1)

    R = cos * eye + sin * u_tilde + (1-cos) * outer
    return R

def rotation_matrix_from_rpy(rpy):
    rpy = rpy[..., tf.newaxis]
    roll, pitch, yaw = tf.unstack(rpy, axis=-2)
    R = R_x(roll) @ R_y(pitch) @ R_z(yaw)
    return R

def rotation_matrix_from_quaternions(quaternions):
    # extract
    q0 = quaternions[0]
    q1 = quaternions[1]
    q2 = quaternions[2]
    q3 = quaternions[3]
     
    # elements
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = tf.convert_to_tensor([[r00, r01, r02],
                                       [r10, r11, r12],
                                       [r20, r21, r22]])
                            
    return rot_matrix


class UrdfForwardKinematic():
    '''A generic kinematic-class that represents the kinematic of any urdf-file.'''
    def __init__(self, urdf_filepath, order):
        self.filepath = urdf_filepath
        self.order = order
        self.n_joints = len(order)
        self._build()

    def _build(self):
        '''parse the urdf-file and save relevant infos and structures.'''
        # urdf parsing
        urdf_tree = UrdfTree(self.filepath)
        
        # kineamtic chains
        all_backward_paths = urdf_tree.get_backward_paths()
        keys = [path[-1] for path in all_backward_paths]
        self.frame_names = keys
        
        # name to index mapping (key: joint_name, value: index)
        values = range(len(keys))
        #self._name_to_idx_map = dict(zip(keys, values))
        DEFAULT_VALUE = len(all_backward_paths)
        values = range(len(keys))
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
        self._name_to_idx_map = tf.lookup.StaticHashTable(initializer,
                                                        default_value=DEFAULT_VALUE, # todo: this is not clean yet
                                                        name='lookup_name_to_index')

        # kinematic chains (key: joint_name, value: list of indices)
        # note: tensorflow only supports 1-to-1 lookup, therefore we map keys to indices first (1-to-1) and later sclice the data-tensor
        # https://stackoverflow.com/questions/64224051/tf-lookup-statichashtable-with-lists-of-arbitrary-sizes-as-values
        max_len = max([len(path) for path in all_backward_paths])
        values = []
        for path in all_backward_paths:
            path_idx = tf.stack([self._name_to_idx_map[tf.constant(p)] for p in path])
            path_idx_padded = tf.pad(tensor=path_idx, paddings=[[0, (max_len - len(path))]], constant_values=len(all_backward_paths))
            values.append(path_idx_padded)
        
        self.kinematic_chains = tf.stack(values) # data-tensor

        # reordering of q-vector
        # note: the re-ordering is required to ensure that the simuation (e.g. Pybullet) uses the same order of joint-indexing as the urdf-file.
        self._q_reordering = tf.constant([self.order.index(key) if key in self.order else len(self.order) for key in keys], dtype=tf.int32)
        
        # specification of coordinate-frame transformations
        rpy = tf.constant([urdf_tree.get_element_by_name(name=key).rpy for key in keys])
        xyz = tf.constant([urdf_tree.get_element_by_name(name=key).xyz for key in keys])
        R_constant = rotation_matrix_from_rpy(rpy)
        self.T_constant = homogenous_transformation(R=R_constant, t=xyz) 
        self.axis = tf.constant([urdf_tree.get_element_by_name(name=key).axis for key in keys])
        joint_type = tf.constant([urdf_tree.get_element_by_name(name=key).joint_type for key in keys])
        joint_type = joint_type[:, tf.newaxis, tf.newaxis]
        self.is_revolute = tf.cast((joint_type=='revolute'), dtype=tf.float32)
        self.is_prismatic = tf.cast((joint_type=='prismatic'), dtype=tf.float32)
        self.is_fixed = tf.cast((joint_type=='fixed'), dtype=tf.float32)
        # self.all_backward_paths = all_backward_paths

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.float32), # q=[q_1, ..., q_n]
                                  tf.TensorSpec(shape=[], dtype=tf.string)]) # frame
    def forward(self, q, frame):
            q = tf.squeeze(q, axis=0)
            chain_len = len(self.T_constant)
            
            q = tf.concat([q, [0]], axis=-1, name='add_zero_to_q')
            q = tf.gather(q, indices=self._q_reordering, axis=-1, name='q_reordering')
            
            # spatial transformations
            T_fixed = tf.eye(4, batch_shape=[chain_len])
            
            R_revolute = rotation_matrix_from_rotation_vector(
                vec=tf.reshape(self.axis, shape=[-1, 3]),
                angle=tf.reshape(q, shape=[-1]),
            )
            t_revolute = tf.zeros(shape=[chain_len, 3], dtype=tf.float32)
            T_revolute = homogenous_transformation(R_revolute, t_revolute)
            
            R_prismatic = tf.eye(3, batch_shape=[chain_len])
            t_prismatic = tf.expand_dims(q, -1)*self.axis
            T_prismatic = homogenous_transformation(R_prismatic, t_prismatic)
            
            T_variable = self.is_fixed * T_fixed \
                       + self.is_revolute * T_revolute \
                       + self.is_prismatic * T_prismatic
            # note: the current implementation of T_variable creates an unavoidable overhead 
            #       (compare: https://stackoverflow.com/questions/66120879/apply-different-functions-along-one-axis-of-a-tensor)
            T = self.T_constant @ T_variable
            T = tf.concat([T, tf.eye(4, batch_shape=[1])], axis=-3, name='add_unit_transformation')
            
            frame_idx = self._name_to_idx_map[frame]
            kinematic_chain = tf.gather(self.kinematic_chains, indices=frame_idx)
            T_gathered = tf.gather(T, indices=kinematic_chain, axis=-3)
            result = reduce_matrix_prod(T_gathered) # matrix-multiply all transformations along the kinematic chain
            return result[tf.newaxis, ...]
    

    @tf.function(input_signature=[tf.TensorSpec(shape=[1,None], dtype=tf.float32), # q=[q_1, ..., q_n]
                                  tf.TensorSpec(shape=[1,None], dtype=tf.float32), # dq/dt
                                  tf.TensorSpec(shape=[], dtype=tf.string)])     # frame
    def differentiate(self, q, qd, frame):
        q = tf.squeeze(q, axis=0)
        qd = tf.squeeze(qd, axis=0)
        print(f'Building the Graph for Kinematic...')
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_2:
            tape_2.watch(q)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(q)
                x = self.forward(q[tf.newaxis, ...], frame)
                x = tf.reshape(x, shape=[-1])
            # J = tape.batch_jacobian(x,q, experimental_use_pfor=False)
            # xd = tf.linalg.matvec(J, q)
            xd = jacobian_vector_product(x, q, qd, tape)
        J = tape.jacobian(x,q, experimental_use_pfor=False)
        c = jacobian_vector_product(xd, q, qd, tape_2)
        print(f'Finished Building the Graph for Kinematic!')
        x, xd, J, c = x[tf.newaxis, ...], xd[tf.newaxis, ...], J[tf.newaxis, ...], c[tf.newaxis, ...]
        return x, xd, J, c
