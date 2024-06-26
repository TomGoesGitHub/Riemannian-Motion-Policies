import tensorflow as tf

class Datamanager:
    ''''''
    def __init__(self, fkine):
        self.fkine = fkine # todo: add frame_names to fkine as attribute
        
        self.state = {
            frame_name: {
                'pos_on_link_in_base_frame': tf.Variable(initial_value=tf.zeros(shape=[0,3]), shape=[None, 3], dtype=tf.float32),
                'pos_on_obstacle_in_base_frame': tf.Variable(initial_value=tf.zeros(shape=[0,3]), shape=[None, 3], dtype=tf.float32),
                'normal_vec': tf.Variable(initial_value=tf.zeros(shape=[0,3]), shape=[None, 3], dtype=tf.float32),
                'distance': tf.Variable(initial_value=tf.zeros(shape=[0]), shape=[None], dtype=tf.float32),
                'relative_position': tf.Variable(initial_value=tf.zeros(shape=[0,3]), shape=[None, 3], dtype=tf.float32),
                #'relative_orientation': tf.Variable(shape=[None, 3], dtype=tf.float32),}
            } for frame_name in self.fkine.frame_names
        } # to be updated via the update-method
    
    def __getitem__(self, key):
        return self.state[key]

    def update(self, q, distance_data):        
        for frame_name in self.fkine.frame_names:
            if sum([d[0]==frame_name for d in distance_data]) > 0:
                frame_state = self.state[frame_name]
                frame_state['pos_on_link_in_base_frame'].assign(tf.stack([d[1] for d in distance_data if d[0]==frame_name]))
                frame_state['pos_on_obstacle_in_base_frame'].assign(tf.stack([d[2] for d in distance_data if d[0]==frame_name]))
        
        distance_data = [self.preprocess(datapoint, q) for datapoint in distance_data]

        for frame_name in self.fkine.frame_names:
            if sum([d[0]==frame_name for d in distance_data]) > 0:
                frame_state = self.state[frame_name]
                frame_state['normal_vec'].assign(tf.stack([d[1] for d in distance_data if d[0]==frame_name]))
                frame_state['distance'].assign(tf.stack([d[2] for d in distance_data if d[0]==frame_name]))
                frame_state['relative_position'].assign(tf.stack([d[3] for d in distance_data if d[0]==frame_name]))
                #frame_name['relative_orientation'].assign(tf.stack([d[4] for d in distance_data]))
        
    def preprocess(self, datapoint, q):
        frame_name, pos_on_link_in_base_frame, _, normal_vec_in_base_frame, distance, _ = datapoint # unpack
        relative_pos_in_joint_frame = self._get_relative_pos(pos_on_link_in_base_frame, frame_name, q)
        return (frame_name, normal_vec_in_base_frame, distance, relative_pos_in_joint_frame)

    def _get_relative_pos(self, pos_on_link_in_base_frame, frame_name, q):
        '''solve the spatial transformation problem.'''
        T_base_joint = self.fkine.forward(q=tf.constant([q], dtype=tf.float32),
                                  frame=tf.constant(frame_name, dtype=tf.string))[0]
        pos_joint_in_base_frame, R_base_joint = T_base_joint[:3, 3], T_base_joint[:3, :3]
        relative_pos_in_base_frame = pos_on_link_in_base_frame - pos_joint_in_base_frame
        R_joint_base = tf.transpose(R_base_joint)
        relative_pos_in_joint_frame =  tf.linalg.matvec(R_joint_base, relative_pos_in_base_frame)
        return relative_pos_in_joint_frame

