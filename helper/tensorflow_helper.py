import tensorflow as tf

def as_batch(tensors):
    if isinstance(tensors, tuple):
        batched = [t[tf.newaxis, ...] for t in tensors]
    else:
        batched = tensors[tf.newaxis, ...]
    return batched