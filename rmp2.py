from abc import abstractmethod
import tensorflow as tf

from taskmap import IdentityTaskmap

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
    
    #@tf.function
    def evaluate(self, x, xd, *args, **kwargs):
        # x, xd = tf.squeeze(x), tf.squeeze(xd)
        f = self._motion_command(x, xd, *args, **kwargs)
        A = self._metric(x, xd, *args, **kwargs)
        return f, A

class TargetAttractor(RiemannianMotionPolicy):
    def __init__(
        self, goal,
        accel_p_gain, accel_d_gain,
        accel_norm_eps, metric_alpha_length_scale,
        min_metric_alpha, max_metric_scalar, min_metric_scalar,
        proximity_metric_boost_scalar,  proximity_metric_boost_length_scale,
        taskmap, name='attractor'):
        super(TargetAttractor, self).__init__(name, taskmap)

        self.goal = goal
        self.accel_p_gain = accel_p_gain
        self.accel_d_gain = accel_d_gain
        self.accel_norm_eps = accel_norm_eps
        self.metric_alpha_length_scale = metric_alpha_length_scale
        self.min_metric_alpha = min_metric_alpha
        self.max_metric_scalar = max_metric_scalar
        self.min_metric_scalar = min_metric_scalar
        self.proximity_metric_boost_scalar = proximity_metric_boost_scalar
        self.proximity_metric_boost_length_scale = proximity_metric_boost_length_scale

    def _motion_command(self, x, xd):
        delta = self.goal - x
        delta_norm = tf.linalg.norm(delta, axis=1)
        delta_norm = tf.expand_dims(delta_norm, -1)
        soft_delta_norm = tf.maximum(delta_norm, self.accel_norm_eps / 10 * tf.ones_like(delta_norm))
        delta_hat = delta / soft_delta_norm
        xdd = self.accel_p_gain * delta / (delta_norm + self.accel_norm_eps) - self.accel_d_gain * xd
        return xdd
    
    def _metric(self, x, xd, *args, **kwargs):
        batch_size, n_dims = x.shape
        batch_size, n_dims = int(batch_size), int(n_dims)

        delta = self.goal - x
        delta_norm = tf.linalg.norm(delta, axis=1)
        delta_norm = tf.expand_dims(delta_norm, -1)
        soft_delta_norm = tf.maximum(delta_norm, self.accel_norm_eps / 10 * tf.ones_like(delta_norm))
        delta_hat = delta / soft_delta_norm

        eye = tf.eye(n_dims, batch_shape=[batch_size], dtype=tf.float32)
        S = tf.einsum('bi, bj->bij', delta_hat, delta_hat)
        scaled_dist = delta_norm / self.metric_alpha_length_scale
        a = (1. - self.min_metric_alpha) * tf.exp(-.5 * scaled_dist * scaled_dist) + self.min_metric_alpha
        a = tf.expand_dims(a, -1)
        metric = a * self.max_metric_scalar * eye + (1. - a) * self.min_metric_scalar * S

        boost_scaled_dist = delta_norm / self.proximity_metric_boost_length_scale
        boost_a = tf.exp(-.5 * boost_scaled_dist * boost_scaled_dist)
        metric_boost_scalar = boost_a * self.proximity_metric_boost_scalar + (1. - boost_a) * 1.
        metric_boost_scalar = tf.expand_dims(metric_boost_scalar, -1)
        metric = metric_boost_scalar * metric
        return metric
    

class JointVelocityCap(RiemannianMotionPolicy):
    def __init__(
        self, max_velocity, velocity_damping_region, damping_gain,
        metric_weight, name='joint_velocity_cap'):

        super(JointVelocityCap, self).__init__(name, taskmap=IdentityTaskmap())
        self.max_velocity = max_velocity
        self.velocity_damping_region = velocity_damping_region
        self.damping_gain = damping_gain
        self.metric_weight = metric_weight
        self.eps = 1e-6
        self.damped_velocity_cutoff = self.max_velocity - self.velocity_damping_region

    @tf.function
    def evaluate(self, x, xd):
        delta_velocity = tf.abs(xd) - self.damped_velocity_cutoff
        xdd = - tf.abs(self.damping_gain * delta_velocity) * tf.sign(xd)

        clipped_relative_velocity = tf.minimum(delta_velocity, self.velocity_damping_region - self.eps)

        velocity_ratio = clipped_relative_velocity / self.velocity_damping_region
        tf.where(tf.abs(xd) < self.damped_velocity_cutoff, tf.zeros_like(velocity_ratio), velocity_ratio)
        diag = tf.linalg.diag(velocity_ratio ** 2)
        metric = self.metric_weight / (1.0 - diag)

        acceleration = tf.where(tf.abs(xd) < self.damped_velocity_cutoff, tf.zeros_like(xdd), xdd)
        return acceleration, metric


class JointDamping(RiemannianMotionPolicy):
    def __init__(
        self,
        accel_d_gain, metric_scalar, inertia,
        name='joint_damping'):

        super(JointDamping, self).__init__(name=name, taskmap=IdentityTaskmap())
        self.accel_d_gain = accel_d_gain
        self.metric_scalar = metric_scalar
        self.inertia = inertia

    @tf.function
    def evaluate(self, x, xd):
        batch_size, x_shape = x.shape

        xd_norm = tf.norm(xd, axis=1, keepdims=True)
        nonlinear_gain = self.accel_d_gain * xd_norm
        acceleration = -nonlinear_gain * xd
        nonlinear_metric_scalar = self.metric_scalar * xd_norm
        nonlinear_metric_scalar = tf.expand_dims(nonlinear_metric_scalar, -1)
        metric = tf.eye(x_shape, batch_shape=[batch_size]) * (nonlinear_metric_scalar + self.inertia)

        return acceleration, metric
    

class ObstacleAvoidance(RiemannianMotionPolicy):
    def __init__(
        self,
        margin,
        damping_gain,
        damping_std_dev,
        damping_robustness_eps,
        damping_velocity_gate_length_scale,
        repulsion_gain,
        repulsion_std_dev,
        metric_modulation_radius,
        metric_scalar,
        metric_exploder_std_dev,
        metric_exploder_eps,
        taskmap,
        name):

        super(ObstacleAvoidance, self).__init__(name=name, taskmap=taskmap)
        self.margin = margin
        self.damping_gain = damping_gain
        self.damping_std_dev = damping_std_dev
        self.damping_robustness_eps = damping_robustness_eps
        self.damping_velocity_gate_length_scale = damping_velocity_gate_length_scale
        self.repulsion_gain = repulsion_gain
        self.repulsion_std_dev = repulsion_std_dev
        self.metric_modulation_radius = metric_modulation_radius
        self.metric_scalar = metric_scalar
        self.metric_exploder_std_dev = metric_exploder_std_dev
        self.metric_exploder_eps = metric_exploder_eps

    def _smooth_activation_gate(self, x):
        r = self.metric_modulation_radius
        gate = x * x / (r * r) - 2. * x / r + 1.
        gate = tf.where(x > r, tf.zeros_like(gate), gate)
        return gate

    def _length_scale_normalized_repulsion_distance(self, x):
        return x / self.repulsion_std_dev

    def _calc_damping_gain_divisor(self, x):
        z = x / self.damping_std_dev + self.damping_robustness_eps
        return z
    
    @tf.function
    def evaluate(self, x, xd):
        x = x - self.margin
        x = tf.maximum(x, tf.zeros_like(x))
        base_metric = self.metric_scalar / (x / self.metric_exploder_std_dev + self.metric_exploder_eps)
        metric = base_metric * self._smooth_activation_gate(x)
        xdd_repel = self.repulsion_gain * tf.exp(-self._length_scale_normalized_repulsion_distance(x))
        sig = tf.sigmoid(xd / self.damping_velocity_gate_length_scale)
        xdd_damping = -(1. - sig) * self.damping_gain * xd / self._calc_damping_gain_divisor(x)

        accel = xdd_repel + xdd_damping
        metric = tf.where(x > self.metric_modulation_radius, tf.zeros_like(metric), (1 - sig) * metric)
        metric = tf.expand_dims(metric, -1)
        return accel, metric
    
class CSpaceBiasing(RiemannianMotionPolicy):
    """
    configuration space target reaching (default configuration)
    """
    def __init__(self, goal, metric_scalar, position_gain, damping_gain,
        robust_position_term_thresh, inertia, taskmap=IdentityTaskmap(), name='cspace_target'):
        super(CSpaceBiasing, self).__init__(name=name, taskmap=taskmap)
        self.goal = goal 
        self.metric_scalar = metric_scalar
        self.position_gain = position_gain
        self.damping_gain = damping_gain
        self.robust_position_term_thresh = robust_position_term_thresh
        self.inertia = inertia

    def evaluate(self, x, xd, **features):
        x = x - self.goal
        batch_size, x_shape = x.shape

        x_hat, x_norm = tf.linalg.normalize(x, axis=1)
        qdd_position = tf.where(
            x_norm < self.robust_position_term_thresh,
            -x * self.position_gain,
            -self.robust_position_term_thresh * x_hat * self.position_gain,
            )
        qdd_velocity = -self.damping_gain * xd
        eye = tf.eye(x_shape, batch_shape=[batch_size])
        metric = eye * (self.metric_scalar + self.inertia)
        acceleration = qdd_position + qdd_velocity
        return acceleration, metric