import pybullet as p

def get_motor_joints(id):
    n_joints = p.getNumJoints(id)
    mask_controlable = [p.getJointInfo(id, i)[3] > -1 for i in range(n_joints)]
    return mask_controlable

def get_joint_order(id, motor_joints_only=False, by='name'):
    assert by in ['name', 'index']
    order = [] # to be filled
    n_joints = p.getNumJoints(id)
    for i in range(n_joints):
        joint_info = p.getJointInfo(id, i)
        joint_index, joint_name = joint_info[0], joint_info[1].decode('ascii')
        is_controllable = joint_info[3] > -1
        if motor_joints_only and (not is_controllable):
            continue
        order.append(joint_name if by=='name' else joint_index)
    return order

def get_kinematic_chains(id, target):
    if isinstance(target, int):
        target = [target]

    kinematic_chains = {}
    for i in target:
        chain = []
        target_name = p.getJointInfo(id, i)[1].decode('ascii')
        
        # barckward pass from target to root
        joint_index = i
        parent_index = None
        while parent_index != (-1):
            # extract info from pybullet
            joint_info = p.getJointInfo(id, joint_index)
            joint_index = joint_info[0]
            joint_name = joint_info[1].decode('ascii')
            parent_index = joint_info[16]
            
            chain.insert(0, joint_name)
            joint_index = parent_index
        
        kinematic_chains[target_name] = chain
    return kinematic_chains

def check_link_neighborhood(robot_id, link_A, link_B, n_neighbors=2):
    if link_A == link_B:
        return True
    
    shared_neighborhood = False
    for link in [link_A, link_B]:
        if shared_neighborhood:
            break
        
        elem = link
        the_other_link = link_B if link is link_A else link_A
        
        for _ in range(n_neighbors):
            if elem == (-1): 
                break # base has no parents
            info = p.getJointInfo(robot_id, elem)
            parent = info[16]
            if parent == the_other_link:
                shared_neighborhood = True
                break
            else:
                elem = parent
    return shared_neighborhood