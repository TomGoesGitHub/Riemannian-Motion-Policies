import tensorflow as tf

def rmp_differentiate(fn):
    tf.function() # todo: TensorSpec
    def differentiate_fn(q, qd):
        # tf.assert_rank(q, 1)
        # tf.assert_rank(qd, 1)
        # q = tf.reshape(q, shape=[1, -1])
        # qd = tf.reshape(qd, shape=[1, -1])
        
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_2:
            tape_2.watch(q)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(q)
                x = fn(q)
            xd = jacobian_vector_product(x, q, qd, tape)
        c = jacobian_vector_product(xd, q, qd, tape_2)

        J = tape.batch_jacobian(x,q, experimental_use_pfor=False)
        #x, xd, J, c = tf.squeeze(x, axis=0), tf.squeeze(xd, axis=0), tf.squeeze(J, axis=0), tf.squeeze(c, axis=0)
        return x, xd, J, c
    return differentiate_fn

# def rmp_differentiate(fn, **kwargs):
#     tf.function() # todo: TensorSpec
#     def differentiate_fn(q, qd):
#         tf.assert_greater(tf.rank(q), 0)
#         # tf.assert_rank(q, 2)
#         # tf.assert_rank(qd, 2)
#         #batch_size = tf.shape(q)[0]
#         # q = tf.reshape(q, shape=[batch_size, -1])
#         # qd = tf.reshape(qd, shape=[batch_size, -1])
#         with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_2:
#             tape_2.watch(q)
#             with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
#                 tape.watch(q)
#                 x = fn(q, **kwargs)
#                 #x = tf.reshape(x, shape=[batch_size, -1])
#             # J = tape.batch_jacobian(x,q, experimental_use_pfor=False)
#             # xd = tf.linalg.matvec(J, q)
#             xd = jacobian_vector_product(x, q, qd, tape)
#         if tf.rank(x)>=2:
#             J = tape.batch_jacobian(x,q, experimental_use_pfor=False)
#         else: # rank=1
#             J = tape.jacobian(x,q, experimental_use_pfor=False)
#         c = jacobian_vector_product(xd, q, qd, tape_2)
#         return x, xd, J, c
#     return differentiate_fn

def jacobian_vector_product(v,u,w, external_tape):
    '''jacobian-Vector-Product'''
    dummy_ones = tf.ones_like(v) #tf.ones(v.shape[-1])
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(dummy_ones)
        with external_tape:
            inner = tf.einsum('...i, ...i -> ...', v, dummy_ones)
        g = external_tape.gradient(target=inner, sources=u, unconnected_gradients='zero') # sum of partial derivatives
        inner2 = tf.einsum('...i, ...i -> ...', g, w)
    jvp = tape.gradient(target=inner2, sources=dummy_ones, unconnected_gradients='zero')
    return jvp

def soft_norm(v, c):
    v_l2norm = tf.norm(v, axis=-1)
    h = v_l2norm + 1/c * tf.math.log(1+tf.exp(-2 * c * v_l2norm))
    return v / h[:, tf.newaxis]

def directionally_stretched_metric(v, beta, c):
    zeta = soft_norm(v, c)
    # A_stretched = tf.einsum('i,j->ij', zeta, zeta) # outer product
    # A_stretched = tf.tensordot(zeta, zeta, axes=0) # outer product
    A_stretched = tf.einsum('...i, ...j -> ...ij', zeta, zeta)
    I = tf.eye(*A_stretched.shape[-2:], batch_shape=[A_stretched.shape[0]])
    H = beta * A_stretched + (1-beta) * I
    return H
