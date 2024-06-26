from xml.etree import ElementTree

class UrdfElem():
    def __init__(self, name, link_name, joint_type='fixed', rpy=[0.,0.,0.], xyz=[0.,0.,0.], axis=[0.,0.,0.], has_collision=True, id=None):
        self.parent = None
        self.children = []
        self.id = id # to be filled by Urdf-Tree
        
        self.name = name # joint name
        self.link_name = link_name
        self.has_collision = has_collision
        self.joint_type = joint_type
        self.rpy = rpy
        self.xyz = xyz
        self.axis = axis
        self.kinematic_chain = []# todo

    def add_child(self, elem):
        assert type(elem) == UrdfElem
        elem.parent = self
        self.children.append(elem)
    
    def get_root(self):
        node = self
        while not node.is_root():
            node = node.parent
        return node
    
    def is_root(self):
        return self.parent==None
    
    def __str__(self):
        str = '\n'*2
        for key, value in self.__dict__.items():
            if type(value) == UrdfElem:
                value = value.__repr__()
            str += f'\n{key} : {value}'
        return str

class UrdfTree():
    def __init__(self, filepath):
        self.filepath = filepath
        
        self.root = None
        self.highest_assigned_id = -1
        self.n_elems = 0

        self._build()

    def create_element(self, name, link_name, joint_type='fixed', rpy=[0.,0.,0.], xyz=[0.,0.,0.], axis=[0.,0.,0.], has_collision=True):
        self.highest_assigned_id += 1
        self.n_elems += 1
        id = self.highest_assigned_id
        elem = UrdfElem(name, link_name, joint_type=joint_type, rpy=rpy, xyz=xyz, axis=axis, id=id, has_collision=has_collision)
        return elem

    def _build(self):
        tree = ElementTree.parse(self.filepath)
        root = tree.getroot()
        all_links = root.findall('link')
        all_joints = root.findall('joint')
        
        # find base link (which is not child of another joint)
        for link in all_links:
            is_root = True
            for joint in all_joints:
                if joint.find('child').attrib['link'] == link.attrib['name']:
                    is_root = False
            if is_root:
                base_link = link
                break
        root_elem = self.create_element(name='<ROOT>', link_name=base_link.attrib['name'])
        self.root = root_elem
        
        todo_list = []
        todo_list.append(root_elem)
        # keep searching in remaining joints and links and append to tree if possible
        while len(todo_list) > 0:
            leaf = todo_list[0]
            for joint in all_joints:
                if joint.find('parent').attrib['link'] == leaf.link_name:
                    l = None
                    for link in all_links:
                        if joint.find('child').attrib['link'] == link.attrib['name']:
                            l = link
                            break
                    elem = self.create_element(name=joint.attrib['name'],
                                               link_name=l.attrib['name'],
                                               joint_type=joint.attrib['type'],
                                               rpy=[float(i) for i in ' '.join(joint.find('origin').attrib['rpy'].split()).split(' ')],
                                               xyz=[float(i) for i in ' '.join(joint.find('origin').attrib['xyz'].split()).split(' ')],
                                               axis=([float(i) for i in ' '.join(joint.find('axis').attrib['xyz'].split()).split(' ')] if (joint.attrib['type'] not in ['fixed']) else [0.,0.,0.] ),
                                               has_collision=True if link.find('collision') else False)
                    leaf.add_child(elem)
                    todo_list.append(elem)
            index = todo_list.index(leaf)
            todo_list.pop(index)
    
    def show(self):
        self.root.show_all()

    def get_element_by_name(self, name):
        if self.root.name == name:
            return self.root
        return self._get_element_by_name_rek(name, self.root)

    def _get_element_by_name_rek(self, name, elem):
        result = None
        for child in elem.children:
            if child.name == name:
                result = child
                break
            result = self._get_element_by_name_rek(name, child)
            if result is not None:
                break
        return result

    def get_element_by_id(self, id):
        if self.root.id == id:
            return self.root
        return self._get_element_by_id_rek(id, self.root)

    def _get_element_by_id_rek(self, id, elem):
        result = None
        for child in elem.children:
            if child.id == id:
                result = child
                break
            result = self._get_element_by_id_rek(id, child)
            if result is not None:
                break
        return result
    
    def get_backward_paths(self, by_name=True):
        all_backward_paths = []
        for id in range(self.n_elems):
            path = []
            start_elem = self.get_element_by_id(id)
            if not start_elem.is_root():
                elem = start_elem
                while not elem.is_root():
                    value = elem.name if by_name else elem.id
                    path.insert(0, value)
                    elem = elem.parent
                #path.insert(0, elem.id) # root
                all_backward_paths.append(path)
        return all_backward_paths
    
    def show(self):
        print('\n')
        self._show_rek(self.root, indents=0)

    def _show_rek(self, elem, indents):
        print('    '*indents, elem.name)
        indents += 1
        for child in elem.children:
            self._show_rek(child, indents)