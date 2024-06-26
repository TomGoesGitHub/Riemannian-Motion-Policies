from abc import abstractmethod
import tensorflow as tf
import numpy as np # todo: numpy or tensorflow?

from helper.rmp_helper import directionally_stretched_metric
from kinematics import R_x, R_y, R_z
from taskmap import IdentityTaskmap


class _RmpCore:
    '''Manages multiple single RMPs. Solves an optimization problem by combining their
    motion commands into one optimal overall command.'''
    def __init__(self, rmps={}):
        # logic
        self.rmps = rmps # key: name, value: RiemannianMotionPolicy

        # state
        self.q = None
        self.qd = None
        self.distances = None
    
    def __str__(self):
        if len(self.rmps) > 0:
            print('\nused RMPs:')
            for i, rmp in enumerate(self.rmps):
                print('\t'.join(i, rmp.name, type(rmp)))
        else:
            print('no RMPs in use.')
    
    def add_rmp(self, rmp):
        self.rmps[rmp.name] = rmp
    
    def remove_rmp_by_name(self, name):
        self.rmps.pop(name)
    
    def update(self, q, qd, distances):
        self.q = q
        self.qd = qd
        self.distances = distances
    
    def evaluate(self):
        # unresolved form (to be incremented)
        n_joints = 9 # todo: hardcoded
        f_combined = np.zeros(shape=n_joints)
        M_combined = np.zeros(shape=(n_joints, n_joints))
        
        q = tf.Variable(self.q, trainable=False, dtype=tf.float32)
        qd = tf.Variable(self.qd, trainable=False, dtype=tf.float32)
        
        for rmp in self.rmps.values():
            if isinstance(rmp, CollisionAvoidance):
                for distance in self.distances:
                    # unpack
                    joint_name, pos_on_link_in_base_frame, normal_vec_in_base_frame, d = distance 
                    
                    
                    pos_on_link_in_base_frame = tf.constant(pos_on_link_in_base_frame, dtype=tf.float32)
                    normal_vec_in_base_frame =  tf.constant(normal_vec_in_base_frame, dtype=tf.float32)
                    
                    # change state of the rmp dynamically
                    rmp.reference_frame = joint_name 
                    rmp.d = d
                    rmp.vec = normal_vec_in_base_frame

                    # spatial transformation
                    T_base_joint = self.fkine(q=tf.constant(q), frame=tf.constant(joint_name))
                    pos_joint_in_base_frame, R_base_joint = T_base_joint[:3, 3], T_base_joint[:3, :3]
                    relative_pos_in_base_frame = pos_on_link_in_base_frame - pos_joint_in_base_frame
                    R_joint_base = tf.transpose(R_base_joint)
                    relative_pos_in_joint_frame =  tf.linalg.matvec(R_joint_base, relative_pos_in_base_frame)

                    # calculate rmp
                    relative_pos = tf.Variable(relative_pos_in_joint_frame, trainable=False, dtype=tf.float32)
                    relative_orn_euler = tf.Variable([0,0,0], trainable=False, dtype=tf.float32)
                    f, M = self._calculate_rmp(rmp, q, qd, tf.constant(rmp.reference_frame), relative_pos, relative_orn_euler)
                    
                    # increment
                    f_combined += tf.linalg.matvec(M,f)
                    M_combined += M

            else:
                relative_pos = tf.Variable([0,0,0], trainable=False, dtype=tf.float32)
                relative_orn_euler = tf.Variable([0,0,0], trainable=False, dtype=tf.float32)
                rmp_kwargs = {}
        
                # calculate rmp
                f, M = self._calculate_rmp(rmp, q, qd, tf.constant(rmp.reference_frame), relative_pos, relative_orn_euler, *rmp_kwargs)
                # increment
                f_combined += tf.linalg.matvec(M,f)
                M_combined += M

        # resolve
        M_combined_pinv = tf.linalg.pinv(M_combined)
        qdd_des = tf.linalg.matvec(M_combined_pinv, f_combined)
        return qdd_des

    def _calculate_rmp(self, rmp, q, qd, relative_pos, relative_orn_euler, *args, **kwargs):
        # forward pass
        x, xd, c, J = rmp.forward(q, qd, relative_pos, relative_orn_euler)
        
        # evalutation of leaf node
        xdd_des, M_leaf  = rmp.evaluate(x, xd, *args, *kwargs) # resolved form
        f_leaf = tf.linalg.matvec(M_leaf, (xdd_des-c)) # todo

        # pullback
        J_transpose = tf.transpose(J)
        M = J_transpose @ M_leaf @ J
        f = tf.linalg.matvec((J_transpose @ M_leaf), f_leaf)
        return f, M # (unresolved form)

class RmpCore:
    '''Manages multiple RMPs. Solves an optimization problem by combining their
    motion commands into one optimal overall command.'''
    def __init__(self, rmps={}):
        self.rmps = rmps # key: name, value: RiemannianMotionPolicy
    
    def __str__(self):
        out_str = ''
        if len(self.rmps) > 0:
            out_str += '\n' + 'used RMPs:' + '\n'
            for i, rmp in enumerate(self.rmps.values()):
                out_str += '\t'.join([str(i), rmp.name, str(type(rmp))]) + '\n'
        else:
            out_str += 'no RMPs in use.' + '\n'
        return out_str
    
    def add_rmp(self, rmp):
        self.rmps[rmp.name] = rmp
    
    def remove_rmp_by_name(self, name):
        self.rmps.pop(name)
    
    def evaluate(self, q, qd):
        # unresolved form (to be incremented)
        n_joints = len(q) # todo: is this ok?
        f_combined = np.zeros(shape=n_joints)
        M_combined = np.zeros(shape=(n_joints, n_joints))
        
        q = tf.Variable(q, trainable=False, dtype=tf.float32)
        qd = tf.Variable(qd, trainable=False, dtype=tf.float32)
        
        for rmp in self.rmps.values():        
            f, M = self._calculate_rmp(rmp, q, qd)
            #f = tf.linalg.matvec(M,f)
            # if tf.rank(f) == 2:
            #     f = tf.reduce_sum(f, axis=0)
            # if tf.rank(M) == 3:
            #     M = tf.reduce_sum(M, axis=0)
            f_combined += tf.reduce_sum(f, axis=0) #tf.linalg.matvec(M,f)
            M_combined += tf.reduce_sum(M, axis=0)

        # resolve
        M_combined_pinv = tf.linalg.pinv(M_combined)
        qdd_des = tf.linalg.matvec(M_combined_pinv, f_combined)
        return qdd_des

    def _calculate_rmp(self, rmp, q, qd):
        # forward pass
        x, xd, J, c = rmp.taskmap.differentiate(q[tf.newaxis, :], qd[tf.newaxis, :])
        
        # evalutation of leaf node
        xdd_des, M_leaf  = rmp.evaluate(x, xd) # resolved form
        
 
        J_transpose = tf.transpose(J, perm=[0,2,1])
        f = tf.linalg.matvec(J_transpose @ M_leaf, xdd_des-c)
        M = J_transpose @ M_leaf @ J

        #f_leaf = tf.linalg.matvec(M_leaf, (xdd_des-c)) # todo

        # # pullback in resolved form
        # J_transpose = tf.transpose(J, perm=[0,2,1])
        # M = J_transpose @ M_leaf @ J
        # f = tf.linalg.matvec(tf.linalg.pinv(M) @ J_transpose @ M_leaf, f_leaf)

        # # pullback
        # J_transpose = tf.transpose(J, perm=[0,2,1])
        # M = J_transpose @ M_leaf @ J
        # f = tf.linalg.matvec((J_transpose @ M_leaf), f_leaf)
        return f, M # (unresolved form)



class RiemannianMotionPolicy:
    '''Abstract class for a single RMP.'''
    def __init__(self, name, taskmap):
        self.name = name
        self.taskmap = taskmap
    
    def _taskmap(self, q):
        '''Task-map'''
        raise NotImplementedError 

    @abstractmethod
    def _motion_command(self, x, xd, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _metric(self, x, xd, *args, **kwargs):
        raise NotImplementedError
    
    def evaluate(self, x, xd, *args, **kwargs):
        # x, xd = tf.squeeze(x), tf.squeeze(xd)
        f = self._motion_command(x, xd, *args, **kwargs)
        A = self._metric(x, xd, *args, **kwargs)
        return f, A

# class RmpWithRelativeDefinition(RiemannianMotionPolicy):
#     def __init__(self, name, reference_mapping, relative_pos, relative_orn_euler):
#         super().__init__(name, reference_mapping)
#         self.relative_pos = relative_pos
#         self.relative_orn = relative_orn_euler
    
#     def _taskmap_4x4(self, q):
#         # mapping from base to reference frame
#         T_0_reference = self.mapping(q)
        
#         # mapping from reference frame to target frame        
#         alpha_x, alpha_y, alpha_z = tf.unstack(self.relative_orn_euler, axis=-1)
#         R = R_x(angle=alpha_x) @ R_y(angle=alpha_y) @ R_z(angle=alpha_z)
#         T_reference_targetframe = homogenous_transformation(R, t=self.relative_pos)
#         T = T_0_reference @ T_reference_targetframe
#         return T


class TargetPolicy(RiemannianMotionPolicy):
    '''RMP that tries to move a certain reference-frame into a certain goal configuration.'''
    def __init__(self, alpha, beta, c, goal, taskmap, name='Target_RMP'):
        super(TargetPolicy, self).__init__(name, taskmap)
        self.goal = goal
        
        # parameters for motion command
        self.c = c
        self.alpha = alpha
        self.beta = beta
        
        # parameters for metric
        self.sigma_H = 1
        self.sigma_w = 3

    def _motion_command(self, x, xd, *args, **kwargs):
        v = self.goal - x
        v_l2norm = tf.norm(v)
        h = v_l2norm + self.c * tf.math.log(1+tf.exp(-2 * self.c * v_l2norm))
        s = 1/h * v
        xdd_des = self.alpha * s - self.beta * xd
        return xdd_des

    def _metric(self, x, xd, *args, **kwargs):
        # directional stretching
        f_attract = self._motion_command(x, xd)
        v_l2norm = tf.norm(x-self.goal)
        beta = 1 - tf.math.exp(-0.5* (v_l2norm)**2 / (self.sigma_H**2))
        H = directionally_stretched_metric(v=f_attract, c=self.c, beta=beta)
        
        # weight function
        w = tf.math.exp(-v_l2norm / self.sigma_w)

        A = w * H
        return A
        # return tf.eye(tf.shape(x)[-1], batch_shape=[tf.shape(x)[0]])


class CollisionAvoidance(RiemannianMotionPolicy):
    '''RMP that tries to avoid nearby obstacles.'''
    def __init__(self, d, vec, eta_rep, nu_rep, eta_damp, nu_damp, r, c, taskmap, name='collision_avoidance'):
        super(CollisionAvoidance, self).__init__(name, taskmap)
        # distance data
        self.d = d
        self.vec = vec
        
        # parameters for motion command
        self.eta_rep = eta_rep
        self.nu_rep = nu_rep
        self.eta_damp = eta_damp
        self.nu_damp = nu_damp

        # parameters for metric
        self.r = r
        self.c = c

    #@tf.function
    def _motion_command(self, x, xd):
        # repulsive term
        alpha_rep = self.eta_rep * tf.exp(-self.d / self.nu_rep)
        f_rep = alpha_rep[:, tf.newaxis] * self.vec
        
        # damping term
        epsilon = tf.constant(1e-6, dtype=tf.float32) # for numerical stability
        alpha_damp = self.eta_damp / (self.d / self.nu_damp + epsilon)
        scaling = tf.math.maximum(0., tf.einsum('...i, ...i -> ...', -xd, self.vec)) # directional scaling
        P_obs = tf.einsum('..., ...i, ...j -> ...ij', scaling, self.vec, self.vec) # outer product
        f_damp = alpha_damp[:, tf.newaxis] * tf.linalg.matvec(P_obs, xd)
        
        f = f_rep - f_damp
        return f
    
    #@tf.function
    def _metric(self, x, xd):
        # weight function (cubic spline with w(0)=1, w'(0)=0, w(r)=0, w'(r)=0)
        c_0 = 1
        c_1 = 0
        c_2 = -3 / self.r**2
        c_3 = 2 / self.r**3
        spline = c_3* self.d**3 + c_2 * self.d**2 + c_1 * self.d + c_0
        w = tf.where(self.d>self.r, tf.zeros_like(spline), spline) #w = tf.reduce_max([spline, tf.zeros_like(spline)], axis=0)
        
        # w = (self.d / self.r)**(-1)

        # directional stretching
        f_obs = self._motion_command(x, xd)
        H = directionally_stretched_metric(v=f_obs, c=self.c, beta=0) # todo: why does beta default to 0 in paper?
        
        A = w[:, tf.newaxis, tf.newaxis] * H
        return A


class ConfigurationSpaceBiasing(RiemannianMotionPolicy):
    '''RMP that tries to move to a certain configuration q0, implemented as a PD-controller.'''
    def __init__(self, gamma_p, gamma_d, q0, name, w=0.05):
        super().__init__(name, taskmap=IdentityTaskmap())
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.q_0 = q0
        self.w = w

        # self.sigma_H = np.pi/2
        # self.c = 0.1
    
    def _motion_command(self, x, xd):
        q, qd = x, xd # rename coordinates for clarification
        xdd_des = self.gamma_p * (self.q_0 - q) - self.gamma_d * qd
        return xdd_des
    
    def _metric(self, x, xd):
        q, qd = x, xd # rename coordinates for clarification
        # f_attract = self._motion_command(q, qd)
        # v_l2norm = tf.norm(q-self.q0)
        # beta = 1 - tf.math.exp(-0.5* (v_l2norm)**2 / (self.sigma_H**2))
        # H = directionally_stretched_metric(v=f_attract, c=self.c, beta=beta)
        
        # # weight function
        # w = tf.math.exp(v_l2norm / self.sigma_w)

        # A = w * H
        # return A
        return self.w * tf.eye(tf.shape(q)[-1], batch_shape=[1])

class JointLimitAvoidance(RiemannianMotionPolicy):
    def __init__(self, lower_limits, upper_limits, gamma_p, gamma_d, name='joint_limit_avoidance'):
        super().__init__(name, taskmap=IdentityTaskmap())
        self.lower_limits = tf.constant(lower_limits, dtype=tf.float32)
        self.upper_limits = tf.constant(upper_limits, dtype=tf.float32)
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        
    def _metric(self, q, qd):
        d_upper = (self.upper_limits - q) / (self.upper_limits-self.lower_limits)
        d_lower = (q - self.lower_limits) / (self.upper_limits-self.lower_limits)
        d = tf.reduce_min([d_upper, d_lower], axis=0)

        
        # weight function (cubic spline with w(0)=1, w'(0)=0, w(r)=0, w'(r)=0)
        self.r = 0.15
        c_0 = 1
        c_1 = 0
        c_2 = -3 / self.r**2
        c_3 = 2 / self.r**3
        spline = c_3* d**3 + c_2 * d**2 + c_1 * d + c_0
        w = tf.where(d>self.r, tf.zeros_like(spline), spline)

        #w = 1/10 * d**-1 

        qd_max = 20*(2*np.pi)/60 # 20 rpm
        v = qd / qd_max
        H = directionally_stretched_metric(v, beta=0.9, c=5)
        A = w * H
        return A
    
    def _motion_command(self, q, qd):
        qdd_des = - self.gamma_p * q - self.gamma_d * qd
        return qdd_des



        
# class JointLimitAvoidanceOld(RiemannianMotionPolicy):
#     '''RMP that tries to avoid the joint limits of robot.'''
#     def __init__(self, lower_limits, upper_limits, gamma_p, gamma_d, lamda, c, name='joint_limit_avoidance'):
#         super().__init__(name, taskmap=IdentityTaskmap())
#         # joint limits
#         self.n_joints = len(lower_limits)
#         self.lower_limits = tf.constant(lower_limits, dtype=tf.float32)
#         self.upper_limits = tf.constant(upper_limits, dtype=tf.float32)
#         assert len(self.lower_limits) == len(self.upper_limits)

#         # parameters
#         self.lamda = lamda 
#         self.c = c
#         self.gamma_p = gamma_p
#         self.gamma_d = gamma_d
    
#     def _motion_command(self, u, ud):
#         udd_des = - self.gamma_p * u - self.gamma_d * ud
#         return udd_des
    
#     def _metric(self, u, ud):
#         I = tf.eye(len(u))
#         A_u = self.lamda * I
#         return A_u

#     def evaluate(self, x, xd):
#         q, qd = x, xd # rename coordinates for clarification
       
#         # pullback into unconstraint space
#         with tf.GradientTape(persistent=True) as tape:
#             tape.watch(q)
#             u = tf.math.log((q-self.lower_limits)/(self.upper_limits-q))
        
#         # jacobian
#         D = tape.jacobian(u, q)
#         diag_elems = tf.linalg.diag_part(D)
#         ones = tf.ones_like(diag_elems)
#         sigma = 1 / (1 + tf.exp(-u))
#         alpha = 1 / (1 + tf.exp(-self.c * qd))
#         diag_elems_tilde = sigma * (alpha * diag_elems + (1-alpha) * ones) \
#                          + (1-sigma) * ((1-alpha) * diag_elems + alpha * ones)
#         D_tilde = tf.linalg.diag(diag_elems_tilde)
#         D_tilde_inverse = tf.linalg.diag(tf.math.reciprocal(diag_elems_tilde)) # inverse of diagonal matrix
#         ud = tf.linalg.matvec(D_tilde_inverse, qd)

#         # evaluate in unconstraint spaces
#         udd_des = self._motion_command(u, ud) # called h in paper
#         A_u = self._metric(u, ud) # A_u = lamda * I
        
#         # push forward into constraint space again
#         diag_elems_A_u = tf.linalg.diag_part(A_u)
#         A_u_inverse = tf.linalg.diag(tf.math.reciprocal(diag_elems_A_u)) # inverse of diagonal matrix
#         qdd_des = tf.linalg.matvec(D_tilde @ A_u_inverse, self.lamda*udd_des)
#         A = self.lamda * D_tilde

#         return qdd_des, A

# class HeuristicLongRangeArmNavigation(RiemannianMotionPolicy):
#     pass # todo


        


