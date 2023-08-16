import numpy as np
S_AB_D=0.2
S_shedding_factor=0.01
GBA_MIN=0.0
GBA_MAX=0.2

'''
    There are some differences between the parameter names here and in the article
    We mark the differences by commented code
'''
class parameter:
    '''some default parameters'''
    AAV = 3 #IAV
    TB_D = 0 #Death probability of a terminal bud
    LF_A = 1.0
    LF_L = 1.05
    AD_DF = 0.5 #Apical dominance distance factor
    AD_AF = 0.9 #Apical dominance age factor
    DM_F = 2.5 #allometric factor
    L_FE = 2.0
    v_light_w = 1.0
    max_resouce_factor = 1.5
    BAV = 5
    RAV = 5
    S_a = 0.5  # Shadow factor a
    S_b = 2  # Shadow factor b
    paramters_dir = None
    paramters_dir_inv=None
    NLB = 0
    BAM = 1
    RAM = 2
    AB_D = 3 #LBD
    AD_BF = 4
    AD_LF = 5
    AC = 6
    AC_AF = 7
    AC_LF = 8
    LB_PF = 9 #PF
    BDM = 10 #1/BWF
    t = 11
    LRR = 12
    shedding_factor = 13 #SF
    v_tropism_w = 14 #PT
    GBF = 15
    GBA = 16
    IL_LF = 17
    FP_Y = 18  # BF
    FP_AF = 19
    FP_A = 20
    MAX_L=21 #ML
    IL_AF=22
    IBL=23 #IL
    LRR_AF = 24
    light_sensitiveness = 25 #LS
    variables_list=None
    variables_limits=None

    data_transport_time = 0
    data_computation_time = 0

    interval = 30.0
    bins_num = round(180.0 / interval)
    branch_angle_bins_num = round(90 / interval)
    branch_angle_bins_weight=None
    bins_weight=None
    light_intensity=None
    TPB = 64
    def __init__(self,variables_list_array=None):

        self.paramters_dir={}
        self.paramters_dir[0] = 'NLB'
        self.paramters_dir[1] = 'BAM'
        self.paramters_dir[2] = 'RAM'
        self.paramters_dir[3] = 'AB_D'
        self.paramters_dir[4] = 'AD_BF'
        self.paramters_dir[5] = 'AD_LF'
        self.paramters_dir[6] = 'AC'
        self.paramters_dir[7] = 'AC_AF'
        self.paramters_dir[8] = 'AC_LF'
        self.paramters_dir[9] = 'LB_PF'
        self.paramters_dir[10] = 'BDM'
        self.paramters_dir[11] = 't'
        self.paramters_dir[12] = 'LRR'
        self.paramters_dir[13] = 'shedding_factor'
        self.paramters_dir[14] = 'v_tropism_w'
        self.paramters_dir[15] = 'GBF'
        self.paramters_dir[16] = 'GBA'
        self.paramters_dir[17] = 'IL_LF'
        self.paramters_dir[18] = 'FP_Y'
        self.paramters_dir[19] = 'FP_AF'
        self.paramters_dir[20] = 'FP_A'
        self.paramters_dir[21]='MAX_L'
        self.paramters_dir[22]='IL_AF'
        self.paramters_dir[23]='IBL'
        self.paramters_dir[24]='LRR_AF'
        self.paramters_dir[25] = 'light_sensitiveness'
        #light_sensitiveness = 0.2
        self.paramters_dir_inv=dict(zip(self.paramters_dir.values(),self.paramters_dir.keys()))
        self.variables_list = np.zeros(len(self.paramters_dir))
        self.variables_limits = np.zeros([len(self.variables_list), 3])  # min max bins(1 means it is int)


        self.variables_limits[self.NLB] = (1, 1, 1)
        self.variables_limits[self.BAM] = (15, 75, 2)
        self.variables_limits[self.RAM] = (30, 150, 2)
        self.variables_limits[self.AB_D] = (0.2, 0.3, 2)
        self.variables_limits[self.AD_BF] = (0.2, 4.0, 2)
        self.variables_limits[self.AD_LF] = (0.5, 1.5, 2)
        self.variables_limits[self.AC] = (0.25, 0.75, 2)
        self.variables_limits[self.AC_AF] = (0.95, 1.05, 2)
        self.variables_limits[self.AC_LF] = (0.8, 1.2, 2)
        self.variables_limits[self.LB_PF] = (0.05, 0.35, 2)
        self.variables_limits[self.BDM] = (40, 40, 1)
        self.variables_limits[self.t] = (12, 18, 2)
        self.variables_limits[self.LRR] = (1.5, 2.5, 2)
        self.variables_limits[self.shedding_factor] = (S_shedding_factor, S_shedding_factor, 1)
        self.variables_limits[self.v_tropism_w] = (0.05, 0.05, 1)
        self.variables_limits[self.GBF] = (10.0, 10.0, 1)
        self.variables_limits[self.GBA] = (0.1, 0.1, 1)
        self.variables_limits[self.IL_LF] = (0.75, 1.0, 2)
        #self.variables_limits[self.FP_Y] = (0.25, 1.25, 2)
        self.variables_limits[self.FP_Y] = (0.25, 0.55, 2)
        self.variables_limits[self.FP_AF] = (1.0, 1.0, 1)
        self.variables_limits[self.FP_A] = (15, 15, 1)
        self.variables_limits[self.MAX_L] = (3, 5, 2)
        self.variables_limits[self.IL_AF] = (0.94, 1, 2)
        self.variables_limits[self.IBL] = (0.5, 0.5, 1)
        self.variables_limits[self.LRR_AF] = (0.96, 1, 2)
        self.variables_limits[self.light_sensitiveness] = (0.1, 0.1, 1)

        self.branch_angle_bins_weight = np.arange(0, 90, self.interval) / self.interval + 1
        self.bins_weight = np.repeat(self.branch_angle_bins_weight, self.bins_num * 2)
        self.light_intensity = self.branch_angle_bins_weight.sum() * self.bins_num * 2
        if variables_list_array is not None:
            for i in range(self.variables_list.shape[0]):
                self.variables_list[i] = variables_list_array[i]
        else:
            for i in range(self.variables_list.shape[0]):
                self.variables_list[i] = (self.variables_limits[i][0]+self.variables_limits[i][1])/2.

    def set_variables(self,variables_list_array):
        for i in range(self.variables_list.shape[0]):
            self.variables_list[i] = variables_list_array[i]
    def get(self,ID):
        if type(ID)==int:
            return self.variables_list[id]
        if type(ID)==str:
            return self.variables_list[self.paramters_dir_inv[ID]]

    def set(self, ID,val):
        if type(ID) == int:
            self.variables_list[id]=val
        if type(ID) == str:
            self.variables_list[self.paramters_dir_inv[ID]]=val

    def print_variables(self):
        variables_num = len(self.variables_list)
        for i in range(variables_num):
            print(self.paramters_dir[i], '=', self.variables_list[i])

    def get_variables_dir(self):
        dir = {}
        for i in range(len(self.variables_list)):
            dir[self.paramters_dir[i]] = self.variables_list[i]
        return dir
