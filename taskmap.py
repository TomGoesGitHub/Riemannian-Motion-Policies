import tensorflow as tf
from helper.rmp_helper import rmp_differentiate
from kinematics import euler_from_rotation_matrix, homogenous_transformation
from helper.rmp_helper import rmp_differentiate, jacobian_vector_product

class Taskmap:
    def forward(self, q):
        raise NotImplementedError
    
    def differentiate(self, q, qd):
        raise NotImplementedError

class IdentityTaskmap:
    def forward(self, q):
        return q
    
    def differentiate(self, q, qd):
        differentiation_function = rmp_differentiate(self.forward)
        x, xd, J, c = differentiation_function(q,qd)
        return x, xd, J, c

class TaskmapByForwardKinematic:
    def __init__(self, fkine, frame):
        self.fkine = fkine
        self.frame = tf.constant(frame, dtype=tf.string)

    def forward(self, q):
        return self.fkine(q, self.frame)
    
    def differentiate(self, q, qd):
        return self.fkine.differentiate(q, qd, self.frame)

class TaskmapByFunction:
    def __init__(self, forward_fn, differentiate_fn):
        self.forward_fn = forward_fn
        self.differentiate_fn = differentiate_fn

    def forward(self, q):
        return self.forward_fn(q)
    
    def differentiate(self, q, qd):
        return self.differentiate_fn(q, qd)


class TaskmapFrom4x4ToPosition:
    def forward(self, input):
        T = tf.reshape(input, shape=[-1,4,4])
        pos = T[:, :3, 3]
        return pos
    
    def differentiate(self, q, qd):
        differentiation_function = rmp_differentiate(self.forward)
        x, xd, J, c = differentiation_function(q,qd)
        return x, xd, J, c


class TaskmapFrom4x4ToEuler:
    def forward(self, input):
        T = tf.reshape(input, shape=[-1,4,4])
        R = T[:, :3, :3]
        eulers = euler_from_rotation_matrix(R)
        return eulers
    
    def differentiate(self, q, qd):
        differentiation_function = rmp_differentiate(self.forward)
        x, xd, J, c = differentiation_function(q,qd)
        return x, xd, J, c


class TaskmapFrom4x4ToQuaternions:
    def forward(self, input):
        raise NotImplementedError # todo
    
    def differentiate(self, q, qd):
        differentiation_function = rmp_differentiate(self.forward)
        x, xd, J, c = differentiation_function(q,qd)
        return x, xd, J, c

class TaskmapRelative4x4:
    def __init__(self, relative_pos):
        self.relative_pos = relative_pos # tf.constant or tf.Variable

    def forward(self, input):
        T_reference = tf.reshape(input, shape=[-1,4,4])
        T_reference = tf.broadcast_to(T_reference, shape=[tf.shape(self.relative_pos)[0],4,4])
        batch_size = tf.shape(T_reference)[0]
        T_relative = homogenous_transformation(R=tf.eye(3, batch_shape=[batch_size]),
                                               t=self.relative_pos)
        T = T_reference @ T_relative
        T_flat = tf.reshape(T, [-1, 16])
        return T_flat
    
    @tf.function
    def differentiate(self, q, qd):
        differentiation_function = rmp_differentiate(self.forward)
        q = tf.repeat(q, repeats=tf.shape(self.relative_pos)[0], axis=0)
        qd = tf.repeat(qd, tf.shape(self.relative_pos)[0], axis=0)
        x, xd, J, c = differentiation_function(q,qd)
        return x, xd, J, c
        
        # tf.assert_rank(q, 2)
        # tf.assert_equal(tf.shape(q)[0], 1) # todo: into tensorspec
        # with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_2:
        #     tape_2.watch(q)
        #     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        #         tape.watch(q)
        #         x = self.forward(q)
        #     xd = jacobian_vector_product(x, q, qd, tape)
        # c = jacobian_vector_product(xd, q, qd, tape_2)

        # J = tape.jacobian(x,q, experimental_use_pfor=False)
        # J = tf.squeeze(J, axis=-2)
        # return x, xd, J, c

class TaskmapJointFrame4x4ToDistance():
    def __init__(self, pos_on_link_in_base_frame, pos_on_obstacle_in_base_frame):
        self.pos_on_link_in_base_frame = pos_on_link_in_base_frame
        self.pos_on_obstacle_in_base_frame = pos_on_obstacle_in_base_frame

    def forward(self, input):
        T_reference = tf.reshape(input, shape=[-1,4,4])
        T_reference = tf.broadcast_to(T_reference, shape=[tf.shape(self.pos_on_link_in_base_frame)[0],4,4])

        pos_joint_in_base_frame_differentiable = T_reference[:, :3, 3]
        relative_pos_in_base_frame = self.pos_on_link_in_base_frame - pos_joint_in_base_frame_differentiable
        relative_pos_in_base_frame = tf.stop_gradient(relative_pos_in_base_frame)

        critical_pos_in_base_frame = pos_joint_in_base_frame_differentiable + relative_pos_in_base_frame
        distance = tf.norm((critical_pos_in_base_frame - self.pos_on_obstacle_in_base_frame), axis=-1)
        tf.debugging.assert_near(distance, tf.norm(self.pos_on_link_in_base_frame-self.pos_on_obstacle_in_base_frame, axis=-1))
        return distance[:, tf.newaxis]
    
    def differentiate(self, q, qd):
        differentiation_function = rmp_differentiate(self.forward)
        q = tf.repeat(q, repeats=tf.shape(self.pos_on_link_in_base_frame)[0], axis=0)
        qd = tf.repeat(qd, tf.shape(self.pos_on_link_in_base_frame)[0], axis=0)
        x, xd, J, c = differentiation_function(q,qd)
        return x, xd, J, c



def _chain_taskmaps(taskmap_1, taskmap_2):
    #@tf.function
    def combined_forward(q):
        out_1 = taskmap_1.forward(q)
        out_2 = taskmap_2.forward(out_1)
        return out_2
    
    #@tf.function
    def combined_differentiate(q, qd):
        out_1, dout1_dt, J_1, c_1  = taskmap_1.differentiate(q, qd)
        # out_1 = tf.reshape(out_1, [-1, out_1.shape[1]])
        # dout1_dt = tf.
        out_2, dout2_dt, J_2, c_2  = taskmap_2.differentiate(out_1, dout1_dt)

        out = out_2
        dout_dt = tf.linalg.matvec(J_2, dout1_dt)
        J = J_2 @ J_1 # tf.linalg.matmul(J_2, J_1)
        c = c_2 + tf.linalg.matvec(J_2, c_1)
        return out, dout_dt, J, c

    return TaskmapByFunction(combined_forward, combined_differentiate)

def chain_taskmaps(taskmap_list):
    chained = taskmap_list[0]
    for taskmap in taskmap_list[1:]:
        chained = _chain_taskmaps(chained, taskmap)
    return chained