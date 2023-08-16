import numpy as np

class Mynode:

    id=-1
    type=-1
    layer = -1
    width = 0
    light = 0.0
    resource = 0.0
    phyllotaxy = 0.0
    rotation = 0.0
    pos = None
    created_age = -1
    left_child = None
    right_child = None
    last_node = None
    cumu_res = 0.0  # cumulative resources
    forking_flag = False
    left_l=0.
    right_l=0
    left_dir=None
    right_dir=None
    left_dir_default=None
    right_dir_deault=None
    left_dir_x=None
    right_dir_x = None
    left_rel_roll_branch=None
    right_rel_roll_branch=None
    afford_weight = 0
    supported_branches_centre=None
    bending_angle=0
    # frame=None
    # frame_r=None

    #left_pruning=False
    #right_pruning=False
    def __init__(self):
        self.pos = np.array([0, 0, 0], dtype=float)
    def reset_node(self):
        #self.id = -1
        self.type = -1
        self.layer = -1
        self.width = 0
        self.light = 0.0
        self.resource = 0.0
        self.phyllotaxy = 0.0
        self.rotation = 0.0
        self.pos = np.array([0, 0, 0], dtype=float)
        self.created_age = -1
        self.left_child = None
        self.right_child = None
        self.last_node = None
        self.cumu_res = 0.0  # cumulative resources
        self.forking_flag = False
        self.left_l = 0.
        self.right_l = 0
        self.left_dir = None
        self.right_dir = None
        self.left_dir_default = None
        self.right_dir_deault = None
        self.left_rel_roll_branch = None
        self.right_rel_roll_branch = None
        self.afford_weight=0
        self.supported_branches_centre = None
        self.bending_angle=0
        self.left_dir_x = None
        self.right_dir_x = None
