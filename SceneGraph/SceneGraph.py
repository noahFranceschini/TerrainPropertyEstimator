import numpy as np
from klampt.math import se3
from collections import deque
class BaseGraphNode:
    r"""
    A base class node for a scene graph of an environment.

    Attributes:
        - name (str): Name of the node
        - transform (R,t): The transform relative to the parent node
                           using the klampt (R,t) transform representation
                           where R is a 9x1 list of a flattened rotation
                           matrix and t is a 3x1 list of a flattened
                           translation matrix.
        - children (list): list of children nodes
    """

    def __init__(self, name, transform, parents=None):
        self.name = name
        self.transform = transform
        self.children = []

    def change_name(self, name: str):
        self.name = name
    
    def change_transform(self, T):
        self.transform = T

    def get_children(self) -> list:
        return self.children
    
    def add_child(self, child: 'BaseGraphNode'):
        self.children.append()

    def get_name(self) -> str:
        return self.name    
    
    def get_children_names(self) -> list[str]:
        r"""
        Returns a list of the names of this node's
        children
        """
        names = []
        for child in self.children:
            names.append(child.name)
        return names

class SceneGraph:
    r"""
    Maintains a tree of GraphNodes, representing
    the world that a robot has observed.

    Attributes:
        - root (BaseGraphNode): the root node of the 
                                SceneGraph
        - nodes (dict): maps node names in graph -> BaseGraphNode
    """

    def __init__(self):
        self.root = None 
        self.nodes = {} 

    def add_node(self,parent_name: str, child_node: 'BaseGraphNode'):
        r"""
        Adds a new node as a child of the specified
        parent node

        :param parent_name: name of parent node
        :param child_node: a pre-filled BaseGraphNode
                           to be added as a child
        """

        #TODO: handle duplicate names
        if parent_name not in self.nodes:
            raise KeyError(f"Parent node: '{parent_name}' not in SceneGraph.")
        
        parent_node = self.nodes[parent_name]
        parent_node.add_child(child_node)
        self.nodes[child_node.name] = child_node

    def add_root(self, root_node: 'BaseGraphNode', override=False):
        r"""
        Adds a new root node if one does not
        already exist

        :param root_node: a pre-filled BaseGraphNode to
                          be added as the root
        """

        if self.root is not None and not override:
            raise ValueError(f"SceneGraph already has a root node. Set override=True if you want to replace the root.")
        elif self.root is not None and override: #overrides current root node and makes old one child of new one
            root_node.add_child(self.root)
        
        self.root = root_node
        self.nodes[root_node.name] = root_node

    def get_node_transformation(self, node_name: str) -> tuple:
        r"""
        Returns the rigid transformation (R,t) of the
        target node relative to it's parent node
        """
        return self.nodes[node_name].transform
    
    def is_in_subtree_of(self, root_name: str, descendant_name: str) -> bool:
        r"""
        Returns True if the descendant is in the subtree of
        the root node

        :param root_node: root node of subtree
        :param descedant: descendant node to check
        """
        root_node = self.nodes[root_name]

        child_queue = deque()
        child_queue.extend(root_node.children)

        # perform BFS on root_node to find this node in subtree
        while len(child_queue) != 0:
            curr_node = child_queue.popleft()

            if curr_node.name == descendant_name: # child found
                return True
            else:
                curr_node_children = curr_node.children

                if len(curr_node_children) != 0:
                    child_queue.extend(curr_node_children)

        # failed to find descendant node in parent node subtree
        return False
    
    def change_node_name(self, target_name: str, new_name: str):
        r"""
        Changes the name of target node to a new name. Also
        changes its name in the node dictionary
        """
        target_node = self.nodes[target_name]
        self.nodes[new_name] = self.nodes[target_name] # replaces old key with new name
        target_node.change_name(target_name)

    def is_parent_of(self, parent_name: str, child_name: str) -> bool:
        r"""
        Returns True if this node is the parent of the
        child_node

        :param parent_name: name of parent node
        :param child_name: name of child node
        """
        parent_node = self.nodes[parent_name]
        child_names = parent_node.get_children_names()

        return True if child_name in child_names else False
    
    def is_child_of(self, parent_name: str, child_name: str) -> bool:
        r"""
        Returns True if this node is the child of the
        parent_node
        """
        parent_node = self.nodes[parent_name]
        child_names = parent_node.get_children_names()
        return True if child_name in child_names else False
    
    def compute_relative_transform(self, target_frame: str, relative_frame: str):
        r"""
        Computes the transformation from a relative
        frame into the target frame

        :param target_frame: frame of reference to compute
                             relative frame into
        :param relative_frame: frame to be transformed
        """

        # find transformation paths to respective frames
        path_to_target = self._dfs_root_path(target_frame)
        path_to_relative = self._dfs_root_path(relative_frame)

        # find least common transformation
        lca_index = self._lca_index(self,path_to_target,path_to_relative)

        # accumulate transformations for each frame
        ancestor_to_target_transforms = []
        ancestor_to_relative_transforms = []

        for node_name in path_to_target[lca_index:]:
            ancestor_to_target_transforms.append(self.get_node_transformation(node_name))
        for node_name in path_to_relative[lca_index:]:
            ancestor_to_relative_transforms.append(self.get_node_transformation(node_name))
        
        # compute respective frames to common ancestor frame
        target_to_ancestor = self._accumulate_transformations(ancestor_to_target_transforms)
        relative_to_ancestor = self._accumulate_transformations(ancestor_to_relative_transforms)

        # compute relative to target frame
        ancestor_to_target = se3.inv(target_to_ancestor)
        relative_to_target = se3.apply(ancestor_to_target,relative_to_ancestor)

        return relative_to_target
    
    def _accumulate_transformations(self, T_list: list) -> tuple:
        r"""
        Applies a list of transformations to each other
        to accumulate a final transformation
        """           
        T_acc = ([1.0, 0.0, 0.0, 
                  0.0, 1.0, 0.0, 
                  0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0])
        for T in T_list:
            T_acc = se3.apply(T_acc,T)
        return T_acc
    
    def _dfs_root_path(self, target_name: str) -> list[str]:
        r"""
        Performs a DFS on the root node and returns
        the path from root to the target node
        """
        # set up stack and dict to keep track of
        # node parents
        dfs_stack = deque()
        visited_from = {}
        path = []
        nodes = self.nodes()
        # start dfs with root node
        root_name = self.root.get_name()
        dfs_stack.appendleft(root_name)
        visited_from[root_name] = None
        
        # perform dfs to find path between root -> target
        while len(dfs_stack) != 0:
            curr_node_name = dfs_stack.popleft()
            
            # found child
            if curr_node_name == target_name:

                # reconstruct path
                while curr_node_name != None:
                    path.append(curr_node_name)
                    curr_node_name = visited_from[curr_node_name]                   
                path.reverse()
                return path
            
            # add children to search
            curr_children = nodes[curr_node_name].get_children_names()

            for child in curr_children:
                dfs_stack.appendleft(child)
                visited_from[child] = curr_node_name
        
        return []

    def _lca_index(self, path_to_node_1: list[str], path_to_node_2: list[str]) -> int:
        r"""
        Helper function to find the index in a path
        of the lowest common ancestor (LCA) between
        two nodes.
        """
        path_1_len = len(path_to_node_1)
        path_2_len = len(path_to_node_2)

        if len(path_to_node_1) == 0 or len(path_to_node_2) == 0:
            return None
        
        lowest_common_ancestor = -1
        curr_idx = 0
        # iterate across paths until ancestors are different
        while curr_idx < min(path_1_len,path_2_len):
            
            if path_to_node_1[curr_idx] == path_to_node_2[curr_idx]:
                lowest_common_ancestor = curr_idx
                curr_idx += 1 
            else:
                break
        return lowest_common_ancestor
    
    def _lca(self, node_1: str, node_2: str) -> 'BaseGraphNode':
        r"""
        Helper function to find the lowest common
        ancestor (LCA) between this node and
        another node. Used primarily in finding
        transformations from one frame to another.
        Returns None if LCA does not exists, implying that
        nodes are from different trees or a node doesn't exist
        """
        # find paths to each node
        path_to_node_1 = self._dfs_root_path(node_1)
        path_to_node_2 = self._dfs_root_path(node_2)
        
        lowest_common_ancestor = self._lca_index(path_to_node_1, path_to_node_2)

        return self.nodes[path_to_node_1[lowest_common_ancestor]]
    

