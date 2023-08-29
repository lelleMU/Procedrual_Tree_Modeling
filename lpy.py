
import numpy as np
from math import *
import time
from numba import cuda, float32
import logging
import os
import Utility
import config
import cv2
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')
TPB = 1024
NB=128
MAX_NODES_SIZE=204800

class BBox:  #voxel grids
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    min_z = None
    max_z = None
    def __init__(self):
        self.min_x=9999
        self.max_x=-9999
        self.min_y=9999
        self.max_y=-9999
        self.min_z=9999
        self.max_z=-9999
    def reset(self):
        self.min_x = 9999
        self.max_x = -9999
        self.min_y = 9999
        self.max_y = -9999
        self.min_z = 9999
        self.max_z = -9999
    def max_r(self):
        x=max(abs(self.min_x),abs(self.max_x))
        y=max(abs(self.min_y),abs(self.max_y))
        return sqrt(x*x+y*y)





@cuda.jit
def CUDA_compute_light_fast(buds_light_array_d, buds_pos_array_d, ls, buds_size, interval,branch_angle_bins_num,bins_num):
    thres_idx = cuda.grid(1)
    bud_idx = int(thres_idx / buds_size)
    target_bud_idx = thres_idx % buds_size
    x, y, z, n = buds_pos_array_d[bud_idx]
    x2, y2, z2, n2 = buds_pos_array_d[target_bud_idx]
    if z2 <= z:
        return
    dx = x2 - x
    dy = y2 - y
    dz = z2 - z
    r = sqrt(dx * dx + dy * dy + dz * dz)
    branch_angle = asin(dz / r) / pi * 180
    roll_angle = atan2(dy, dx) / pi * 180
    branch_angle_bin = int(branch_angle / interval) if dz != r else branch_angle_bins_num - 1
    roll_angle_bin = (int(roll_angle / interval) + bins_num) % (2 * bins_num)
    bin_index = roll_angle_bin * 3 + branch_angle_bin
    # if buds_light_array_d[buds_size,bin_index]==0.0:
    #     return
    cuda.atomic.sub(buds_light_array_d, (bud_idx, bin_index), n2 * ls)
    cuda.atomic.max(buds_light_array_d, (bud_idx, bin_index), 0)

def CPU_compute_light_fast(buds_light_array, buds_pos_array, ls, buds_size, interval=30.,branch_angle_bins_num=3,bins_num=6):
    for bud_idx in range(buds_size):
        for target_bud_idx in range(buds_size):
            x, y, z, n = buds_pos_array[bud_idx]
            x2, y2, z2, n2 = buds_pos_array[target_bud_idx]
            if z2 <= z:
                continue
            dx = x2 - x
            dy = y2 - y
            dz = z2 - z
            r = sqrt(dx * dx + dy * dy + dz * dz)
            branch_angle = asin(dz / r) / pi * 180
            roll_angle = atan2(dy, dx) / pi * 180
            branch_angle_bin = int(branch_angle / interval) if dz != r else branch_angle_bins_num - 1
            roll_angle_bin = (int(roll_angle / interval) + bins_num) % (2 * bins_num)
            bin_index = roll_angle_bin * 3 + branch_angle_bin
            buds_light_array[bud_idx, bin_index] = max(0, buds_light_array[bud_idx, bin_index] - n2 * ls)

############################################## angle computation #################################################

def compute_angels_from_dir(dir):
    try:
        branch_angle = asin(min(dir[2], 1.0)) / pi * 180
        roll_angle = atan2(dir[1], dir[0]) / pi * 180
        return roll_angle, branch_angle
    except:
        print(dir)
        return 0., 90.

def compute_Rotation_Matrix(r, b):
    r = r / 180.0 * pi
    b = b / 180.0 * pi
    Mz = np.identity(3)
    Mz[0, 0] = cos(r)
    Mz[0, 1] = -sin(r)
    Mz[1, 0] = sin(r)
    Mz[1, 1] = cos(r)
    My = np.identity(3)
    My[0, 0] = sin(b)
    My[0, 2] = cos(b)
    My[2, 0] = -cos(b)
    My[2, 2] = sin(b)
    return Mz.dot(My)

def compute_Rotation_Matrix_from_dir(dir):
    r, b = compute_angels_from_dir(dir)
    return compute_Rotation_Matrix(r, b)

def compute_Rotation_Matrix_xyz(x, y, z):
    x = x / 180.0 * pi
    y = y / 180.0 * pi
    z = z / 180. * pi
    Mz = np.identity(3)
    Mz[0, 0] = cos(z)
    Mz[0, 1] = -sin(z)
    Mz[1, 0] = sin(z)
    Mz[1, 1] = cos(z)
    Mx = np.identity(3)
    Mx[1, 1] = cos(x)
    Mx[1, 2] = -sin(x)
    Mx[2, 1] = sin(x)
    Mx[2, 2] = cos(x)
    My = np.identity(3)
    My[0, 0] = sin(y)
    My[0, 2] = cos(y)
    My[2, 0] = -cos(y)
    My[2, 2] = sin(y)
    return (Mz.dot(Mx)).dot(My)

def Rotation_Matrix_from_y_rotation(angle):  # from z-axis to x-axis angle is positive [x,y,z].dot(M) p.dot(M) trans to R_frame
    M = np.eye(3, dtype=np.float32)
    M[0, 0] = cos(angle * pi / 180.)
    M[0, 2] = -sin(angle * pi / 180.)
    M[2, 0] = sin(angle * pi / 180.)
    M[2, 2] = cos(angle * pi / 180.)
    return M.transpose()

def Rotation_Matrix_from_x_rotation(angle):  # from y-axis to z-axis angle is positive [x,y,z].dot(M)
    M = np.eye(3, dtype=np.float32)
    M[1, 1] = cos(angle * pi / 180.)
    M[1, 2] = sin(angle * pi / 180.)
    M[2, 1] = -sin(angle * pi / 180.)
    M[2, 2] = cos(angle * pi / 180.)
    return M.transpose()

def Rotation_Matrix_from_z_rotation(angle):  # from x-axis to y-axis angle is positive [x,y,z].dot(M)
    M = np.eye(3, dtype=np.float32)
    M[0, 0] = cos(angle * pi / 180.)
    M[0, 1] = sin(angle * pi / 180.)
    M[1, 0] = -sin(angle * pi / 180.)
    M[1, 1] = cos(angle * pi / 180.)
    return M.transpose()

def compute_dir2(roll_old, branch_old, roll, branch):
    M1 = compute_Rotation_Matrix(roll_old, branch_old)
    M2 = compute_Rotation_Matrix(roll, branch)
    M = M1.dot(M2)
    dir = M[:, 2]
    return dir

def compute_dir(last_inter_dir, rel_roll, rel_branch):
    old_roll, old_branch = compute_angels_from_dir(last_inter_dir)
    return compute_dir2(old_roll, old_branch, rel_roll, rel_branch)
def compute_dir_last_dirzx_roll_branch(last_dir_z,last_dir_x,roll,branch):
    dir_y=np.cross(last_dir_z,last_dir_x)
    dir_x=np.cross(dir_y,last_dir_z)
    M1=np.eye(3)
    M1[:,0]=dir_x
    M1[:,1]=dir_y
    M1[:,2]=last_dir_z
    M=M1.dot(compute_Rotation_Matrix(roll, branch))
    return M[:,2],M[:,0]


def compute_RM_xyz(last_inter_dir, x, y, z):
    old_roll, old_branch = compute_angels_from_dir(last_inter_dir)
    M1 = compute_Rotation_Matrix(old_roll, old_branch)
    M2 = compute_Rotation_Matrix_xyz(x, y, z)
    return M1.dot(M2)

def direction_from_angles(roll, branch):
    vz = sin(branch / 180.0 * pi)
    vx = cos(branch / 180.0 * pi) * cos(roll / 180.0 * pi)
    vy = cos(branch / 180.0 * pi) * sin(roll / 180.0 * pi)
    dir = np.array([vx, vy, vz], dtype=float)
    return dir

def compute_rel_angles(dir1, dir2):
    roll, branch = compute_angels_from_dir(dir1)
    M = compute_Rotation_Matrix(roll, branch)
    dirx = M[:, 0]
    diry = M[:, 1]
    z = dir1.dot(dir2)
    x = dirx.dot(dir2)
    y = diry.dot(dir2)
    return compute_angels_from_dir((x, y, z))
############################################## tree structrue graph ###################################

def construct_graph(n1, n2):
    # node<-->node
    left_flag=True if n1.left_child is None else False
    if left_flag:
        n1.left_child=n2.id
        n2.last_node=n1.id
    else:
        n1.right_child = n2.id
        n2.last_node = n1.id

##################################### pointcloud guide direction computation ##############################
def compute_guide_dir(cloud_array, pos, default_dir, l_thres):
    dxyz = cloud_array - pos
    dis_array = np.sqrt((dxyz * dxyz).sum(1))
    norm_dir = dxyz / dis_array.reshape([dis_array.shape[0], 1])
    angle_dot = (norm_dir * default_dir).sum(1)
    flag_angle = angle_dot > 0.866
    if flag_angle.sum() == 0:
        #print('return')
        return 100, np.array([0, 0, 0]), 100
    dis_mean = np.mean(dis_array[flag_angle])
    flag_dis = dis_array < l_thres
    flag = flag_angle * flag_dis
    valid_dis_array = dis_array[flag]
    valid_norm_dir = norm_dir[flag]
    if valid_dis_array.shape[0] == 0:
        return 50, np.array([0, 0, 0]), dis_mean
    valid_dis_mean = np.mean(valid_dis_array)
    w = -valid_dis_array / l_thres
    w = np.exp(w)
    dir_w_mean = (valid_norm_dir * w.reshape([w.shape[0], 1])).sum(0) / (w.sum())
    dir_w_mean = dir_w_mean / sqrt(dir_w_mean.dot(dir_w_mean.transpose()))
    if isnan(dir_w_mean[0]):
        a = 0
    return valid_dis_mean, dir_w_mean, dis_mean

def compute_cloest_dis_from_pc(cloud_array, pos, default_dir, angle_thres=30):
    dxyz = cloud_array - pos
    dis_array = np.sqrt((dxyz * dxyz).sum(1))
    norm_dir = dxyz / dis_array.reshape([dis_array.shape[0], 1])
    angle_dot = (norm_dir * default_dir).sum(1)
    flag = angle_dot > cos(angle_thres / 180 * pi)
    if flag.sum() == 0:
        return -100
    valid_dis_array = dis_array[flag]
    return valid_dis_array.min()


def project_to_image(ts,camera_pos,focal,resolution):
    '''return valid pixel position'''
    dts=ts-camera_pos
    x=dts[:,0]/dts[:,1]*focal*resolution
    y=dts[:,2]/dts[:,1]*focal*resolution
    #y=h-y
    return np.hstack((x.reshape(x.shape[0],1),y.reshape(y.shape[0],1)))


#active_buds_pos_array=np.zeros([2400,4],dtype=float)
@cuda.jit
def CUDA_ray_casting(vs_d,vts_d,f_vs_d,f_vts_d,f_n_d,f_pixel_boundary_d,textures0_d,textures1_d,bd,M_d,M_depth_d,resolution,focal,camera_pos,w,h,x_c,y_c):
    '''f_pixel_boundary_d:(x_min,y_min,x_max,y_max)'''
    #idx=cuda.blockIdx*TPB+cuda.threadIdx
    idx=cuda.grid(1)
    if idx>=w*h:
        return
    x=int(idx%w)
    y=int(idx/w)
    pixel_pos_x = (x + 0.5 - x_c) / resolution
    pixel_pos_z = (y_c - (y + 0.5)) / resolution
    f_num=f_vs_d.shape[0]
    pixel_pos = (pixel_pos_x + camera_pos[0], focal + camera_pos[1], pixel_pos_z + camera_pos[2])
    dir = (pixel_pos[0] - camera_pos[0], pixel_pos[1] - camera_pos[1], pixel_pos[2] - camera_pos[2])
    dir=Utility.CUDA_normalize_3d(dir)
    #const_textures_d=cuda.const.array_like(textures_d)
    for i in range(f_num):
        boundary = f_pixel_boundary_d[i]
        if x<boundary[0] or y<boundary[1] or x>boundary[2] or y>boundary[3]:
            continue
        p1 = vs_d[f_vs_d[i,0]]
        p2 = vs_d[f_vs_d[i,1]]
        p3 = vs_d[f_vs_d[i,2]]
        u1 = vts_d[f_vts_d[i, 0]]
        u2 = vts_d[f_vts_d[i, 1]]
        u3 = vts_d[f_vts_d[i, 2]]

        v12 = Utility.CUDA_subtraction_3d(p2,p1)
        v23 = Utility.CUDA_subtraction_3d(p3,p2)
        v31 = Utility.CUDA_subtraction_3d(p1,p3)

        if i>=bd:
            texture_d=textures1_d
        else:
            texture_d=textures0_d
        #texture_d=textures_d[t_id_d[i]]
        tex_h=texture_d.shape[0]
        tex_w=texture_d.shape[1]
        plane_para=Utility.to_tuple_3d(f_n_d[i])+Utility.to_tuple_3d(p1)
        line_para=dir+camera_pos
        intersection,depth = Utility.CUDA_get_3d_line_plane_cross(plane_para,line_para)
        if depth==0.0:
            continue
        #depth = Utility.CUDA_distance_3d(intersection, camera_pos)
        if depth > M_depth_d[y,x] and M_d[y,x,3] != 0:
            continue
        cross1 = Utility.CUDA_cross_3d(v12, Utility.CUDA_subtraction_3d(intersection,p1))
        cross2 = Utility.CUDA_cross_3d(v23, Utility.CUDA_subtraction_3d(intersection,p2))
        cross3 = Utility.CUDA_cross_3d(v31, Utility.CUDA_subtraction_3d(intersection,p3))
        if Utility.CUDA_dot_3d(cross1,cross2)>0 and Utility.CUDA_dot_3d(cross1,cross3)>0 and Utility.CUDA_dot_3d(cross2,cross3):
            w1 = Utility.CUDA_3d_length(cross2)
            w2 = Utility.CUDA_3d_length(cross3)
            w3 = Utility.CUDA_3d_length(cross1)
            wu1=Utility.CUDA_multiplication_2d_scalar(u1,w1)
            wu2=Utility.CUDA_multiplication_2d_scalar(u2,w2)
            wu3=Utility.CUDA_multiplication_2d_scalar(u3,w3)
            wup=Utility.CUDA_addition_2d(wu1,wu2)
            wup=Utility.CUDA_addition_2d(wup,wu3)
            up=Utility.CUDA_division_2d_scalar(wup,w1+w2+w3)
            #up = (w1 * u1 + w2 * u2 + w3 * u3) / (w1 + w2 + w3)
            up_x = min(int(up[0] % 1. * tex_w), tex_w - 1)
            up_y = min(int((1 - up[1] % 1.) * tex_h), tex_h - 1)
            if texture_d[up_y, up_x, 3] == 0:
                continue

            M_d[y,x,0]=texture_d[up_y,up_x,0]
            M_d[y, x, 1] = texture_d[up_y, up_x, 1]
            M_d[y, x, 2] = texture_d[up_y, up_x, 2]
            M_d[y, x, 3] = texture_d[up_y, up_x, 3]
            #M_d[y, x] = texture_d[up_y, up_x]
            M_depth_d[y, x] = depth
            #cur_depth=depth


def CUDA_print_image(vs, vts, f_vs, f_vts,R,texture0,texture1,bd,anchor_point,resolution=100000,max_height=2000):
    #start=time.time()
    vs_trans=vs.dot(R)
    #vs_trans=vs
    min_ = vs_trans.min(axis=0)
    max_ = vs_trans.max(axis=0)
    dis = max(max_-min_)
    focal = 0.035
    center=(min_+max_)/2
    camera_pos = center
    camera_pos[1] = min_[1] - dis
    ts_xy = project_to_image(vs_trans, camera_pos, focal, resolution)
    p_max=ts_xy.max(axis=0)
    p_min=ts_xy.min(axis=0)
    w = ceil((p_max[0] - p_min[0]))
    h = ceil((p_max[1] - p_min[1]))
    if h>max_height:
        ratio=max_height/h
        resolution*=ratio
        ts_xy*=ratio
        p_max*=ratio
        p_min*=ratio
        w = ceil((p_max[0] - p_min[0]))
        h = ceil((p_max[1] - p_min[1]))
    trans_anchor_point=anchor_point.dot(R)
    temp_dis=trans_anchor_point[1]-camera_pos[1]
    lb=camera_pos+np.array([p_min[0]/resolution*temp_dis/focal,temp_dis,p_min[1]/resolution*temp_dis/focal])
    lu=camera_pos+np.array([p_min[0]/resolution*temp_dis/focal,temp_dis,p_max[1]/resolution*temp_dis/focal])
    ru = camera_pos + np.array(
        [p_max[0] / resolution * temp_dis / focal, temp_dis, p_max[1] / resolution * temp_dis / focal])
    rb = camera_pos + np.array(
        [p_max[0] / resolution * temp_dis / focal, temp_dis, p_min[1] / resolution * temp_dis / focal])
    four_corners=np.array((lb,lu,ru,rb))
    four_corners=four_corners.dot(np.linalg.inv(R))
    # vs_trans=vs_trans-camera_pos

    x_c = -p_min[0]
    y_c = -p_min[1]
    ts_xy=ts_xy+(x_c,y_c)
    ts_xy[:,1]=h-ts_xy[:,1]
    y_c=h-y_c
    M_depth = np.ones((h, w), float) * 999.
    M = np.zeros((h, w, 4), np.uint8)
    M[:, :, 3] = 0
    cuda.pinned(M)
    f_vs_pos = vs_trans[f_vs]
    v1s = f_vs_pos[:, 1, :] - f_vs_pos[:, 0, :]
    v2s = f_vs_pos[:, 2, :] - f_vs_pos[:, 0, :]
    f_n = np.cross(v1s, v2s)
    f_t = ts_xy[f_vs]
    f_t=f_t.astype('int')
    f_pixel_boundary=np.hstack((f_t.min(axis=1),f_t.max(axis=1)))
    f_pixel_boundary_d=cuda.to_device(f_pixel_boundary)
    M_d=cuda.to_device(M)
    M_depth_d=cuda.to_device(M_depth)

    vs_d = cuda.to_device(vs_trans)
    vts_d=cuda.to_device(vts)
    f_vs_d=cuda.to_device(f_vs)
    f_vts_d=cuda.to_device(f_vts)
    f_n_d=cuda.to_device(f_n)
    texture0_d=cuda.to_device(texture0)
    texture1_d = cuda.to_device(texture1)
    blockspergrid = ceil(w*h / 128)
    threadsperblock = 128
    CUDA_ray_casting[blockspergrid,threadsperblock](vs_d,vts_d,f_vs_d,f_vts_d,f_n_d,f_pixel_boundary_d,texture0_d,texture1_d,bd,M_d,M_depth_d,resolution,focal,tuple(camera_pos),w,h,x_c,y_c)
    M=M_d.copy_to_host()
    return M,four_corners


class tree_scene(object):
    #root=None
    terminals_pos=None
    box=None
    current_age = 0
    vox_size = 0
    buds_cluster_size = 0
    largest_light = 0
    largest_shadow_single_dir = 0
    tropism_dir = None
    phototropism_dir = None
    global_pointcloud = None
    global_height = None
    global_mid_height=None
    max_resource = 0.0
    para=None
    nodes_counts=0
    pool=None
    constrain_flag=False
    def __init__(self,t_para:config.parameter,pool_):
        self.active_buds_pos_table = {}
        self.active_buds_light_table = {}
        self.dis4 = 0
        self.para=t_para
        self.terminals_pos = []
        self.box=BBox()
        self.vox_size = t_para.get('IBL')/1.0
        self.buds_cluster_size = self.vox_size
        self.largest_light=t_para.light_intensity
        self.tropism_dir = np.array([0, 0, 1], dtype=float)
        self.phototropism_dir = np.array([0, 0, 1], dtype=float)
        self.pool = pool_.copy()
        self.pause_flag=False
        self.terminal_flag=False
        self.jump_out_flag=False
        self.pointcloud_array=None
        self.boundary_points=None

    def update_box(self,pos):
        x = pos[0]
        y = pos[1]
        z = pos[2]
        if x < self.box.min_x:
            self.box.min_x = x
        if x > self.box.max_x:
            self.box.max_x = x
        if y < self.box.min_y:
            self.box.min_y = y
        if y > self.box.max_y:
            self.box.max_y = y
        if z < self.box.min_z:
            self.box.min_z = z
        if z > self.box.max_z:
            self.box.max_z = z

    def out_of_box(self,index):
        if index[0] > self.box.max_x or index[0] < self.box.min_x:
            return True
        if index[1] > self.box.max_y or index[1] < self.box.min_y:
            return True
        if index[2] > self.box.max_z or index[2] < self.box.min_z:
            return True
        return False
    def get_active_buds_pos_table(self):
        return self.active_buds_pos_table
    def get_active_buds_light_table(self):
        return self.active_buds_light_table
    def get_dis4(self):
        return self.dis4
    def set_dis4(self,val):
        self.dis4=val
    def get_max_height(self):
        return self.box.max_z
    def set_active_buds_light_table(self,active_buds_light_table):
        #print('set_active_buds_light_table',len(active_buds_light_table),len(self.active_buds_pos_table))
        self.active_buds_light_table=active_buds_light_table
    def get_rescale_global_pointcloud(self,h):
        global_height=self.global_pointcloud.max(0)[2]
        return self.global_pointcloud / global_height*h
    def set_gpc(self,pointcloud_array,height):
        scaled_pointcloud = pointcloud_array * height
        min_h = scaled_pointcloud.min(0)[2]
        self.global_pointcloud = scaled_pointcloud
        self.global_height = height
        self.global_mid_height = (height + min_h) / 2.
    def set_pointcloud(self,pointcloud_array):
        self.pointcloud_array=pointcloud_array
    def set_height(self,height):
        scaled_pointcloud = self.pointcloud_array * height
        min_h = scaled_pointcloud.min(0)[2]
        self.global_pointcloud = scaled_pointcloud
        self.global_height = height
        self.global_mid_height = (height + min_h) / 2.
    def if_pause(self):
        return self.pause_flag
    def if_terminal(self):
        return self.terminal_flag
    def if_jump_out(self):
        return self.jump_out_flag
    def set_pause(self,flag):
        self.pause_flag=flag
        return
    def set_terminal(self,flag):
        self.terminal_flag=flag
        return
    def set_jump_out(self,flag):
        self.jump_out_flag=flag
        return
    def set_para(self,para):
        self.para=para
    ##################################  shadow-light model   ###########################################
    def compute_angular_dis(self,dir):
        return np.dot(self.phototropism_dir, dir)

    def compute_shadow(self):  # create shadow for the new born buds and very recently born buds
        self.active_buds_pos_table.clear()
        #tp=self.root
        stack=[]
        stack.append(0)
        while len(stack)!=0:
            p_id=stack.pop()
            p=self.pool[p_id]
            if p.type == 0 or p.type == 1:
                rand = np.random.rand()
                if p.type == 0:
                    if self.current_age >= 5 and rand < self.para.TB_D:  # need not to create shadows if dead
                        p.type = 4 if p.type == 0 else 3
                    else:
                        pos = p.pos
                        if self.global_pointcloud is not None:
                            if self.constrain_flag==False and pos[2]>=self.global_mid_height:
                                self.constrain_flag=True
                        index = (
                            int(pos[0] / self.buds_cluster_size), int(pos[1] / self.buds_cluster_size),
                            int(pos[2] / self.buds_cluster_size))
                        if index in self.active_buds_pos_table:
                            self.active_buds_pos_table[index] += 1
                        else:
                            self.active_buds_pos_table[index] = 1

                else:
                    if rand < self.para.variables_list[self.para.AB_D] or p.layer>=round(self.para.get('MAX_L')):
                        p.type = 3
                    else:
                        pos = p.pos
                        if isnan(pos[0]):
                            a = 0
                        index = (
                            int(pos[0] / self.buds_cluster_size), int(pos[1] / self.buds_cluster_size),
                            int(pos[2] / self.buds_cluster_size))
                        if index in self.active_buds_pos_table:
                            self.active_buds_pos_table[index] += 1
                        else:
                            self.active_buds_pos_table[index] = 1
            if p.right_child is not None:
                stack.append(p.right_child)
            if p.left_child is not None:
                stack.append(p.left_child)
        if len(self.active_buds_pos_table)==0:
            return False
        else:
            active_buds_light_table=self.EZ_CUDA_compute_light_intensity_fast(self.active_buds_pos_table)
            self.reset_node(active_buds_light_table)  # reset the cumulation to 0
            self.cumu_node()
            return True

    def update_active_buds_pos_table(self):  # create shadow for the new born buds and very recently born buds
        self.active_buds_pos_table.clear()
        #tp=self.root
        stack=[]
        stack.append(0)
        while len(stack)!=0:
            p_id=stack.pop()
            p=self.pool[p_id]
            if p.type == 0 or p.type == 1:
                rand = np.random.rand()
                if p.type == 0:
                    if self.current_age >= 5 and rand < self.para.TB_D:  # need not to create shadows if dead
                        p.type = 4 if p.type == 0 else 3
                    else:
                        pos = p.pos
                        if self.global_pointcloud is not None:
                            if self.constrain_flag==False and pos[2]>=self.global_mid_height:
                                self.constrain_flag=True
                        index = (
                            int(pos[0] / self.buds_cluster_size), int(pos[1] / self.buds_cluster_size),
                            int(pos[2] / self.buds_cluster_size))
                        if index in self.active_buds_pos_table:
                            self.active_buds_pos_table[index] += 1
                        else:
                            self.active_buds_pos_table[index] = 1

                else:
                    if rand < self.para.variables_list[self.para.AB_D] or p.layer>=round(self.para.get('MAX_L')):
                        p.type = 3
                    else:
                        pos = p.pos
                        if isnan(pos[0]):
                            a = 0
                        index = (
                            int(pos[0] / self.buds_cluster_size), int(pos[1] / self.buds_cluster_size),
                            int(pos[2] / self.buds_cluster_size))
                        if index in self.active_buds_pos_table:
                            self.active_buds_pos_table[index] += 1
                        else:
                            self.active_buds_pos_table[index] = 1
            if p.right_child is not None:
                stack.append(p.right_child)
            if p.left_child is not None:
                stack.append(p.left_child)
        #rint('active_buds_pos_table',len(self.active_buds_pos_table))
    def compute_shadow_from_active_buds_light_table(self):
        #self.active_buds_light_table=active_buds_light_table
        self.reset_node(self.active_buds_light_table)  # reset the cumulation to 0
        self.cumu_node()

    def EZ_CUDA_compute_light_intensity_fast(self,active_buds_pos_table):
        size = len(active_buds_pos_table)
        active_buds_light_array = np.ones([size, 36], dtype=np.float32)
        active_buds_pos_array = np.zeros([size, 4], dtype=np.float32)
        i = 0
        for pos in active_buds_pos_table:
            n = active_buds_pos_table[pos]
            x = pos[0]
            y = pos[1]
            z = pos[2]
            active_buds_pos_array[i] = (x, y, z, n)
            i += 1
        if size <= 48:
            CPU_compute_light_fast(active_buds_light_array, active_buds_pos_array, self.para.get('light_sensitiveness'), size)
        else:
            active_buds_light_array_d = cuda.to_device(active_buds_light_array)
            active_buds_pos_array_d = cuda.to_device(active_buds_pos_array)
            threadsperblock = 128
            blockspergrid = ceil(size * size / 128)

            CUDA_compute_light_fast[blockspergrid, threadsperblock](active_buds_light_array_d,
                                                                         active_buds_pos_array_d,
                                                                         self.para.get('light_sensitiveness'), size, 30., 3.,
                                                                         6.)
            active_buds_light_array = active_buds_light_array_d.copy_to_host()
        weights = (1, 2, 3) * 12
        w_array = np.array(weights).reshape(36, 1)
        active_buds_light = active_buds_light_array.dot(w_array)
        active_buds_light_table={}
        for n in range(size):
            x = round(active_buds_pos_array[n][0])
            y = round(active_buds_pos_array[n][1])
            z = round(active_buds_pos_array[n][2])
            active_buds_light_table[(x, y, z)] = active_buds_light[n][0]
        return active_buds_light_table

    def reset_node(self,active_buds_light_table):
        stack = []
        stack.append(0)
        while len(stack) != 0:
            p_id = stack.pop()
            p=self.pool[p_id]
            p.cumu_res = 0
            p.resource = 0
            if p.type == 0 or p.type == 1:
                x = int(p.pos[0] / self.buds_cluster_size)
                y = int(p.pos[1] / self.buds_cluster_size)
                z = int(p.pos[2] / self.buds_cluster_size)
                l = active_buds_light_table[(x, y, z)] / self.largest_light
                # l = EZ3_compute_light_intensity(p.pos)
                p.light = l * self.para.L_FE
            else:
                p.light = 0
            if p.right_child is not None:
                stack.append(p.right_child)
            if p.left_child is not None:
                stack.append(p.left_child)

    def cumu_node(self,p_id=0):
        p=self.pool[p_id]
        # if p is None:
        #     p=self.pool[0]
        res_above = p.light * self.para.variables_list[self.para.LRR] * pow(self.para.get('LRR_AF'), self.current_age)
        if p.left_child is not None:
            res_above+=self.cumu_node(p.left_child)
        if p.right_child is not None:
            res_above += self.cumu_node(p.right_child)
        p.cumu_res = res_above
        return p.cumu_res


    def compute_bud_fate_from_top_to_root(self,root_id):
        root=self.pool[root_id]
        # find top
        p = root
        if p.last_node == None:
            p = self.pool[p.left_child]
            # p = p.next_nodes[0]
        else:
            # p = p.next_nodes[1]
            p = self.pool[p.right_child]
        while p.left_child is not None:
            p = self.pool[p.left_child]
        rand = np.random.rand()
        t_AD_AF = pow(self.para.AD_AF, self.current_age)
        l_AD_LF = pow(self.para.variables_list[self.para.AD_LF], p.layer)
        t_AD_AF *= l_AD_LF
        if rand > pow(p.light / self.para.L_FE, self.para.LF_A):
            # print('this_bud is not flushing',pow(p.light/L_FE,LF_A))
            p.resource = 0.0
            # p.type==4
        bud_above_influnce = 0.0
        while p.last_node != root.id:
            bud_above_influnce *= self.para.AD_DF
            if p.resource > 0.0:
                bud_above_influnce += self.para.variables_list[self.para.AD_BF] * t_AD_AF * self.para.AD_DF
            last_p = self.pool[p.last_node]
            if last_p.type == 1:
                flushing_possibility = pow(last_p.light / self.para.L_FE, self.para.LF_L) * exp(-bud_above_influnce)
                rand = np.random.rand()
                if rand > flushing_possibility:
                    # print('this_bud is not flushing')
                    last_p.resource = 0.0
            elif last_p.type == 2:
                self.compute_bud_fate_from_top_to_root(last_p.id)
            p = last_p
    def compute_bud_fate_from_top_to_root2(self,root_id):
        root=self.pool[root_id]
        # find top
        p = root
        if p.last_node == None:
            p = self.pool[p.left_child]
            # p = p.next_nodes[0]
        else:
            # p = p.next_nodes[1]
            p = self.pool[p.right_child]
        while p.left_child is not None:
            p = self.pool[p.left_child]
        rand = np.random.rand()
        t_AD_AF = pow(self.para.AD_AF, self.current_age)
        l_AD_LF = pow(self.para.variables_list[self.para.AD_LF], p.layer)
        t_AD_AF *= l_AD_LF
        bud_above_influnce = 0.0
        if rand > pow(p.light / self.para.L_FE, self.para.LF_A):
            p.resource = 0.0
        else:
            bud_above_influnce+=self.para.variables_list[self.para.AD_BF] * t_AD_AF
        p=self.pool[p.last_node]
        while p.id != root.id:
            bud_above_influnce *= self.para.AD_DF if p.left_l>0 else 1.0
            if p.type==1:
                flushing_possibility = pow(p.light / self.para.L_FE, self.para.LF_L) * exp(-bud_above_influnce)
                rand = np.random.rand()
                if rand > flushing_possibility:
                    # print('this_bud is not flushing')
                    p.resource = 0.0
                    #print('I am dead!')
                else:
                    bud_above_influnce += self.para.variables_list[self.para.AD_BF] * t_AD_AF
            elif p.type==2:
                bud_above_influnce += self.compute_bud_fate_from_top_to_root2(p.id) * self.para.AD_DF
            p = self.pool[p.last_node]
        return bud_above_influnce
    def distribute_pm(self,p_id=0):
        p=self.pool[p_id]
        # if p is None:
        #     p=self.pool[0]
        if p.cumu_res == 0:
            return
        w_list = []
        p_id_list = []
        tp = p
        cur_AC = self.para.variables_list[self.para.AC] * pow(self.para.variables_list[self.para.AC_AF], self.current_age) * pow(self.para.variables_list[self.para.AC_LF], p.layer)
        while True:
            if tp.type == 2:
                w = self.pool[tp.right_child].cumu_res * (1 - cur_AC)
                w_list.append(w)
                p_id_list.append(tp.id)
            if tp.type == 1:
                p_contribution = tp.light * self.para.variables_list[self.para.LRR] * pow(self.para.get('LRR_AF'), self.current_age)
                w = p_contribution * (1 - cur_AC)
                w_list.append(w)
                p_id_list.append(tp.id)
            if tp.type == 0:
                p_contribution = tp.light * self.para.variables_list[self.para.LRR] * pow(self.para.get('LRR_AF'), self.current_age)
                w = p_contribution * cur_AC
                w_list.append(w)
                p_id_list.append(tp.id)
            if tp.right_child is None and tp.left_child is None:
                break
            tp=self.pool[tp.left_child]
        w_all = sum(w_list)
        if w_all == 0:
            return
        for i in range(len(w_list)):
            tp = self.pool[p_id_list[i]]
            w = w_list[i]
            if tp.type == 0 or tp.type == 1:
                tp.resource = p.cumu_res * w / w_all
            if tp.type == 2:
                self.pool[tp.right_child].cumu_res = p.cumu_res * w / w_all
                self.distribute_pm(tp.right_child)

    ##################################### bending  ##########################################

    def compute_afford_weight(self,p2_id):
        # compute the affording weight for each branch buds
        p2 = self.pool[p2_id]
        if p2.layer == 0:
            if p2.left_child is not None:
                self.compute_afford_weight(p2.left_child)
            if p2.right_child is not None:
                self.compute_afford_weight(p2.right_child)
            return
        p1=self.pool[p2.last_node]
        w1 = p1.width / (self.para.get('IBL') / self.para.variables_list[self.para.BDM])
        w2 = p2.width / (self.para.get('IBL') / self.para.variables_list[self.para.BDM])
        pos1 = p1.pos
        pos2 = p2.pos
        l=p1.left_l if p1.layer==p2.layer else p1.right_l

        w = (w1 + w2) / 2.0 * l
        pos = (pos1 + pos2) / 2.0
        p2.afford_weight = w
        weighted_pos = w * pos
        if p2.left_child is not None:
            w_next, centre_next = self.compute_afford_weight(p2.left_child)
            p2.afford_weight+=w_next
            weighted_pos += w_next * centre_next
        if p2.right_child is not None:
            w_next, centre_next = self.compute_afford_weight(p2.right_child)
            p2.afford_weight += w_next
            weighted_pos += w_next * centre_next

        p2.supported_branches_centre = weighted_pos / p2.afford_weight if p2.afford_weight > 0 else weighted_pos
        return p2.afford_weight, p2.supported_branches_centre

    def compute_bending_angle(self,p2_id):
        p2=self.pool[p2_id]
        p1=self.pool[p2.last_node]
        if p1.afford_weight > 0:
            mc = p1.afford_weight
            fb = mc
            p1.bending_angle = self.para.get('GBF') * fb * pow(self.para.get('GBA'), p1.width)
        if p2.left_child is not None:
            self.compute_bending_angle(p2.left_child)
        if p2.right_child is not None:
            self.compute_bending_angle(p2.right_child)


    def compute_bending_angle2(self,p2_id):
        p2=self.pool[p2_id]
        p1=self.pool[p2.last_node]
        if p1.last_node is not None and (p2.layer>0):
            # update inter direction first
            p0=self.pool[p1.last_node]
            direction=None
            default_direction=None
            if p2_id == p1.left_child:
                p1.left_dir=compute_dir(p0.left_dir,p1.left_rel_roll_branch[0],p1.left_rel_roll_branch[1]) if p0.layer==p1.layer\
                    else compute_dir(p0.right_dir,p1.left_rel_roll_branch[0],p1.left_rel_roll_branch[1])
                direction=p1.left_dir
            else:
                p1.right_dir = compute_dir(p0.left_dir, p1.right_rel_roll_branch[0],
                                          p1.right_rel_roll_branch[1]) if p0.layer == p1.layer \
                    else compute_dir(p0.right_dir, p1.right_rel_roll_branch[0], p1.right_rel_roll_branch[1])
                direction=p1.right_dir

            h = direction

            mc = p2.afford_weight
            g = np.array([0.0, 0.0, -1.0])
            f1 = abs(h.dot(g))
            f2 = sqrt(1 - f1 * f1)
            fb = mc * f2 * self.para.get('GBF') * pow(self.para.get('GBA'), p1.width / (self.para.get('IBL') / self.para.get('BDM')))
            #print(fb)
            roll_angle, branch_angle = compute_angels_from_dir(direction)
            branch_angle = max(-90.0, branch_angle - fb)
            direction=direction_from_angles(roll_angle, branch_angle)
            if p2_id == p1.left_child:
                p1.left_dir = direction
            else:
                p1.right_dir = direction
            length=p1.left_l if p2_id == p1.left_child else p1.right_l

            p2.pos=p1.pos + direction * length
        if p2.left_child is not None:
            self.compute_bending_angle2(p2.left_child)
        if p2.right_child is not None:
            self.compute_bending_angle2(p2.right_child)

    def bending(self):
        self.compute_afford_weight(1)
        self.compute_bending_angle2(1)
    ########################################## width computation ###################################################
    def compute_node_width(self,p_id=0):
        p=self.pool[p_id]
        if p.left_child is not None:
            temp = 0
            next_width = self.compute_node_width(p.left_child)
            temp += pow(next_width, self.para.DM_F)
            if p.right_child is not None:
                next_width = self.compute_node_width(p.right_child)
                temp += pow(next_width, self.para.DM_F)
            if temp != 0:
                p.width = pow(temp, 1 / self.para.DM_F)
            return p.width
        return p.width

    def revise_node_width(self,p_id=0):
        p=self.pool[p_id]
        if p.left_child is None:
            if p.type == 4:
                tp = p
                counts = 0
                while tp.type != 2 and tp.last_node is not None:
                    tp = self.pool[tp.last_node]
                    counts += 1
                    if counts > 100:
                        print('stuck in while!')
                        break
                if tp.last_node is None:
                    return
                if tp.type != 2:
                    print('tp.type!=2')
                if tp.layer == p.layer:
                    if p.width / tp.width < 0.5:
                        self.pool[tp.left_child].type=6
                        self.pool[tp.left_child].left_child=None
                        self.pool[tp.left_child].right_child=None
                        self.terminals_pos.append(tp.pos)
                    else:
                        self.terminals_pos.append(p.pos)
                else:
                    self.terminals_pos.append(p.pos)
            if p.type == 0:
                self.terminals_pos.append(p.pos)
        else:
            if p.left_child is not None:
                self.revise_node_width(p.left_child)
            if p.right_child is not None:
                self.revise_node_width(p.right_child)

    ############################################ pruning and shedding#################################3
    def set_node_dead(self,p_id):
        p=self.pool[p_id]
        p.type = 6
        # node_table.pop(p.id)
        if p.left_child is not None:
            self.set_node_dead(p.left_child)
        if p.right_child is not None:
            self.set_node_dead(p.right_child)

    def pruning_node(self,p_id):
        p=self.pool[p_id]
        p.type = 2
        # self.set_node_dead(p.next_nodes[1])inter

        #self.set_node_dead(self.pool[p.right_child])
        self.pool[p.right_child].type = 6
        self.pool[p.right_child].left_child=None
        self.pool[p.right_child].right_child = None
    def compute_branch_internodes_num(self,p_id):
        p=self.pool[p_id]
        if p.left_child is None:
            return 1
        next_counts = 0
        if p.left_child is not None:
            next_counts += self.compute_branch_internodes_num(p.left_child)
        if p.right_child is not None:
            next_counts += self.compute_branch_internodes_num(p.right_child)
        return next_counts + 1

    def compute_branch_light(self,p_id):
        p=self.pool[p_id]
        if p.left_child is None:
            return p.light,1
        next_nodes_light_all = 0
        next_nodes_counts_all=0
        if p.left_child is not None:
            next_nodes_light,next_nodes_counts=self.compute_branch_light(p.left_child)
            next_nodes_light_all+=next_nodes_light
            next_nodes_counts_all+=next_nodes_counts
            #next_nodes_light += self.compute_branch_light(p.left_child)
        if p.right_child is not None:
            next_nodes_light, next_nodes_counts=self.compute_branch_light(p.right_child)
            next_nodes_light_all += next_nodes_light
            next_nodes_counts_all += next_nodes_counts
            #next_nodes_light += self.compute_branch_light(p.right_child)
        return next_nodes_light_all + p.light,next_nodes_counts_all+1

    def compute_branch_light_active(self, p_id):
        p = self.pool[p_id]
        p_light=p.light
        p_counts=1 if (p.type==0 or p.type==1) else 0
        if p.left_child is None:
            return p_light,p_counts
        next_nodes_light_all = 0
        next_nodes_counts_all = 0
        if p.left_child is not None:
            next_nodes_light, next_nodes_counts = self.compute_branch_light(p.left_child)
            next_nodes_light_all += next_nodes_light
            next_nodes_counts_all += next_nodes_counts
            # next_nodes_light += self.compute_branch_light(p.left_child)
        if p.right_child is not None:
            next_nodes_light, next_nodes_counts = self.compute_branch_light(p.right_child)
            next_nodes_light_all += next_nodes_light
            next_nodes_counts_all += next_nodes_counts
            # next_nodes_light += self.compute_branch_light(p.right_child)
        return next_nodes_light_all + p_light, next_nodes_counts_all + p_counts

    def shedding(self,p_id=0):
        # if p is None:
        #     p=self.pool[0]
        # print('compute for shedding...',p.type)
        # if p.type == 4:
        #     return
        p=self.pool[p_id]
        if p.type == 2 and self.pool[p.right_child].type !=6:
            #light= self.compute_branch_light(p.right_child) / L_FE
            light,counts = self.compute_branch_light_active(p.right_child)
            light/=self.para.L_FE
            #internodes_num = self.compute_branch_internodes_num(p.right_inter)
            # internodes_num = self.compute_branch_internodes_num(p.right_child)
            # print('light',light,'internodes_num',internodes_num)
            ratio = light / counts
            if ratio < self.para.variables_list[self.para.shedding_factor]:
                # print('shedding..',p.id,p.last_node.id)
                self.pruning_node(p.id)
        if p.left_child is not None:
            self.shedding(p.left_child)
        if p.right_child is not None:
            self.shedding(p.right_child)
        # for next in p.next_nodes:
        #     self.shedding(next)

    def pruning(self,p_id, this_LB_PF):
        p=self.pool[p_id]
        if p.left_child is None:
            return
        # first compute for the full length of this level
        l = 0.0
        p_temp = p
        if p.type != 2:
            # l += p_temp.left_inter.length
            l += p_temp.left_l
            #p_temp = p_temp.next_nodes[0]
            p_temp = self.pool[p_temp.left_child]
        else:
            #l += p_temp.right_inter.length
            l += p_temp.right_l
            p_temp = self.pool[p_temp.right_child]
        while p_temp.left_child is not None:
            l += p_temp.left_l
            p_temp = self.pool[p_temp.left_child]
        tl = 0
        tp = self.pool[p.right_child] if p.type == 2 else self.pool[p.left_child]

        while tp.left_child is not None and l > 0:
            #tl += tp.last_inter.length
            tl += Utility.distance(self.pool[tp.last_node].pos,tp.pos)
            if tp.type == 2:  # if a branching point
                tl_ratio = tl / l
                if tl_ratio < this_LB_PF:  # pruning the whole branch
                    self.pruning_node(tp.id)
                else:  # pruning part of the branch using this function
                    next_level_LB_PF = pow(self.para.variables_list[self.para.LB_PF], pow(1 / self.para.variables_list[self.para.LB_PF], tl_ratio))
                    self.pruning(tp.id, next_level_LB_PF)
            tp = self.pool[tp.left_child]

    def dynamic_pruning(self):
        if self.current_age % 3 == 0 and self.current_age > 0:
            self.pruning(0, self.para.variables_list[self.para.LB_PF])



    def compute_bud_fate(self):
        # determine the bud fate according the Apical dominance
        # reset_node(node_table[0])
        if self.current_age > 0:
            self.compute_bud_fate_from_top_to_root2(0)


    ##################################### compute distance ##################################
    def compute_dis(self,cloud_array):
        b_pc = np.array(self.terminals_pos)
        if b_pc.shape[0] == 0:

            return 999., 999., b_pc
        dis_accum = 0
        counts = 0
        for i in range(cloud_array.shape[0]):
            pos = cloud_array[i]
            dxyz = b_pc - pos
            dis_array = np.sqrt((dxyz * dxyz).sum(1))
            min_dis = dis_array.min()
            dis_accum += min_dis
            counts += 1
        dis2 = dis_accum / counts
        return 0, dis2, None

    ##################################### control  ##########################################
    def Start(self):
        # reset
        self.box.reset()
        self.max_resource = self.para.L_FE * self.para.variables_list[self.para.LRR] * self.para.max_resouce_factor

        #p = node()
        p=self.pool[0]
        p.reset_node()
        self.nodes_counts+=1
        p.type = 0
        # p.condition=1
        p.layer = 0
        # p.order=0
        p.created_age = 0
        p.phyllotaxy = 90.0
        p.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
        self.update_box(p.pos)
        self.current_age=0

    def StartEach(self):
        flag=self.compute_shadow()
        if flag==False:
            return False
        self.distribute_pm()
        self.compute_bud_fate()
        self.shedding()
        self.dynamic_pruning()
        return True

    def EndEach(self):
        self.current_age += 1
        self.compute_node_width()
        if self.current_age==round(self.para.get('t')):
            self.bending()
    def End(self,flag=True):
        self.revise_node_width()

    def grow_branch_let_from_lb(self,p):
        branch_angle = np.random.normal(self.para.variables_list[self.para.BAM], self.para.BAV)
        v_default, v_default_x = compute_dir_last_dirzx_roll_branch(self.pool[p.last_node].left_dir,
                                                                    self.pool[p.last_node].left_dir_x, p.rotation,
                                                                    branch_angle)
        roll, branch = compute_angels_from_dir(v_default)
        this_p=p
        for i in range(2):
            v = compute_Rotation_Matrix(roll, max(branch - 15*(i+1),-90))[:, 2]

            if i == 0:
                this_p.right_dir = v
                this_p.right_l = self.para.get('IBL')
            else:
                this_p.left_dir =v
                this_p.left_l = self.para.get('IBL')
            this_p.right_dir = v
            this_p.right_l = self.para.get('IBL')
            lb = self.pool[self.nodes_counts]
            lb.reset_node()
            self.nodes_counts += 1
            # print(self.nodes_counts)
            lb.width = self.para.get('IBL') / self.para.variables_list[self.para.BDM]
            lb.type = 3
            lb.layer = p.layer + 1
            lb.pos = this_p.pos + this_p.right_dir * this_p.right_l if i == 0 else this_p.pos + this_p.left_dir * this_p.left_l
            lb.created_age = self.current_age
            construct_graph(this_p, lb)
            this_p = lb
        this_p.left_dir = self.pool[this_p.last_node].right_dir if this_p.last_node == p.id else self.pool[this_p.last_node].left_dir
        this_p.left_l = 0
        tb = self.pool[self.nodes_counts]
        tb.reset_node()
        self.nodes_counts += 1
        tb.width = self.para.get('IBL') / self.para.variables_list[self.para.BDM]
        tb.type = 4
        tb.layer = this_p.layer
        tb.pos = this_p.pos
        tb.phyllotaxy = 0
        tb.created_age = self.current_age
        construct_graph(this_p, tb)
        self.update_box(tb.pos)
        p.type = 2
    def grow_branch_let(self):
        stack = []
        stack.append(0)
        lb_list = []
        while len(stack) != 0:
            p_id = stack.pop()
            p = self.pool[p_id]
            if (p.type == 3 or p.type==1) and (self.para.get('t')-p.created_age)<=4 and \
                    p.width>(self.para.get('IBL') / self.para.variables_list[self.para.BDM]*1.001):
                lb_list.append(p)
                # self.grow_shot_from_terminal_bud(p)
            if p.right_child is not None:
                stack.append(p.right_child)
            if p.left_child is not None:
                stack.append(p.left_child)

        for lb in lb_list:
            self.grow_branch_let_from_lb(lb)
    def grow_shot_from_terminal_bud(self,tp):
        if self.current_age == round(self.para.variables_list[self.para.t]):
            return
        if tp.type != 0:
            return
        internode_num = min(round(self.max_resource), round(tp.resource))
        forking_flag = False
        forking_possibilty = 0.0
        l_p=tp
        while l_p.layer==tp.layer and l_p.last_node is not None:
            l_p=self.pool[l_p.last_node]
        if (self.current_age-l_p.created_age)==round(self.para.get('t')*self.para.get('FP_Y')):
            forking_flag=True
            forking_possibilty = 1.0
        this_internode_l = self.para.get('IBL') * pow(self.para.get('IL_AF'), self.current_age) * pow(self.para.variables_list[self.para.IL_LF], tp.layer)
        if internode_num == 0:
            tp.type = 4
            return
        terminal_flag = False
        last_apical_roll_angle = 0
        if internode_num > 0:
            p = tp
            apical_roll_angle = np.random.rand() * 360.0
            apical_branch_angle = np.random.normal(90, self.para.AAV)
            v_default = compute_dir(self.pool[p.last_node].left_dir, apical_roll_angle,
                                    apical_branch_angle) if p.last_node != None else self.tropism_dir
            if forking_flag:
                rand = np.random.rand()
                if rand < forking_possibilty:
                    lb1 = self.pool[self.nodes_counts]
                    self.nodes_counts+=1
                    lb1.reset_node()
                    lb1.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
                    lb1.type = 1
                    lb1.layer = p.layer
                    lb1.pos = p.pos
                    lb1.created_age = self.current_age
                    lb1.phyllotaxy = tp.phyllotaxy
                    lb1.rotation = lb1.phyllotaxy
                    lb1.forking_flag = True
                    lb1.resource = self.max_resource
                    p.left_dir=self.pool[p.last_node].left_dir
                    p.left_dir_x=self.pool[p.last_node].left_dir_x
                    p.left_l=0.
                    p.left_rel_roll_branch=(0,90)
                    construct_graph(p, lb1)
                    lb2 = self.pool[self.nodes_counts]
                    lb2.reset_node()
                    self.nodes_counts+=1
                    lb2.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
                    lb2.type = 1
                    lb2.layer = p.layer
                    lb2.pos = p.pos
                    lb2.created_age = self.current_age
                    lb2.phyllotaxy = lb1.phyllotaxy
                    lb2.rotation = lb2.phyllotaxy + np.random.normal(180, 2*self.para.RAV)
                    lb2.forking_flag = True
                    lb2.resource = self.max_resource
                    lb1.left_dir = self.pool[lb1.last_node].left_dir
                    lb1.left_dir_x = self.pool[lb1.last_node].left_dir_x
                    lb1.left_l = 0.
                    lb1.left_rel_roll_branch = (0, 90)
                    construct_graph(lb1, lb2)
                    tb = self.pool[self.nodes_counts]
                    tb.reset_node()
                    self.nodes_counts+=1
                    tb.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
                    tb.type = 4
                    tb.layer = p.layer
                    tb.pos = p.pos
                    tb.phyllotaxy = 0
                    tb.created_age = self.current_age
                    lb2.left_dir = self.pool[lb2.last_node].left_dir
                    lb2.left_dir_x = self.pool[lb2.last_node].left_dir_x
                    lb2.left_l=0.
                    lb2.left_rel_roll_branch = (0, 90)
                    construct_graph(lb2, tb)
                    self.grow_shot_from_lateral_bud(lb1)
                    self.grow_shot_from_lateral_bud(lb2)
                    if tp.last_node is not None and self.pool[tp.last_node].left_l == 0:
                        self.pool[tp.last_node].type = 3
                    if self.pool[tp.last_node].last_node is not None and self.pool[self.pool[tp.last_node].last_node].left_l == 0:
                        self.pool[self.pool[tp.last_node].last_node].type = 3
                    tp.type = 3
                    return
            l_thres = this_internode_l * internode_num
            v_light = v_default
            this_vlight_w = 0.0
            if self.global_pointcloud is not None:
                dis_mean, dir_mean, _ = compute_guide_dir(self.global_pointcloud, p.pos, v_default, l_thres)
                v_light = dir_mean
                if dis_mean == 100:
                    tp.type = 4
                    return

                if dis_mean < l_thres:
                    this_vlight_w = 1.6
            v_tropism = self.tropism_dir
            v_bud = v_default + this_vlight_w * v_light + v_tropism * self.para.variables_list[self.para.v_tropism_w]
            v = v_bud / sqrt(v_bud.dot(v_bud.transpose()))
            for i in range(internode_num):
                this_phyllotaxy = p.phyllotaxy + np.random.normal(self.para.variables_list[self.para.RAM], self.para.RAV)
                direction=None
                if i == 0:
                    direction = v
                else:
                    random_rotation = np.random.rand() * 360.0
                    random_branch_angle = np.random.normal(90, self.para.AAV)
                    last_dir=self.pool[p.last_node].left_dir
                    direction = compute_dir(last_dir, random_rotation,
                                                  random_branch_angle)
                if self.global_pointcloud is not None:
                    c_dis = compute_cloest_dis_from_pc(self.global_pointcloud, p.pos, direction)
                    if c_dis < this_internode_l * 1.5 and self.constrain_flag:
                        terminal_flag = True
                        if c_dis < this_internode_l * 0.5:
                            break
                length = this_internode_l
                p.left_l = length
                p.left_dir = direction
                p.left_dir_x = self.pool[p.last_node].left_dir_x if p.last_node is not None else np.array([1,0,0])
                p.left_rel_roll_branch=compute_rel_angles(self.pool[p.last_node].left_dir,direction) if p.last_node \
                                    is not None else compute_rel_angles(self.tropism_dir,direction)
                this_NLB = 1
                if forking_flag:
                    rand = np.random.rand()
                    if rand < forking_possibilty:
                        this_NLB = 2
                    forking_flag = False
                for n in range(this_NLB):

                    lb = self.pool[self.nodes_counts]
                    lb.reset_node()
                    self.nodes_counts+=1
                    lb.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
                    lb.type = 3 if terminal_flag else 1
                    lb.layer = p.layer
                    lb.pos = p.pos + p.left_dir * p.left_l
                    lb.created_age = self.current_age
                    lb.phyllotaxy = this_phyllotaxy
                    lb.rotation = 360.0 / this_NLB * n + lb.phyllotaxy
                    construct_graph(p, lb)
                    p = lb
                    if n != this_NLB - 1:
                        p.left_l = 0
                        p.left_dir = self.pool[p.last_node].left_dir
                        p.left_dir_x = self.pool[p.last_node].left_dir_x
                        p.left_rel_roll_branch = (0, 90)

            if p.id == tp.id:
                tp.type = 4
                return
            p.left_l = 0
            p.left_dir = self.pool[p.last_node].left_dir
            p.left_dir_x = self.pool[p.last_node].left_dir_x
            p.left_rel_roll_branch = (0, 90)
            tb = self.pool[self.nodes_counts]
            tb.reset_node()
            self.nodes_counts+=1
            tb.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
            tb.type = 4 if terminal_flag else 0
            tb.layer = p.layer
            tb.pos = p.pos
            self.update_box(tb.pos)
            tb.phyllotaxy = 0
            tb.created_age = self.current_age

            construct_graph(p, tb)
            tp.type = 3

    def grow_shot_from_lateral_bud(self,lp):
        if self.current_age == round(self.para.variables_list[self.para.t]):
            return
        if lp.type != 1:
            return
        parent_level = lp.layer
        internode_num = min(round(self.max_resource), round(lp.resource))
        forking_flag = False
        forking_possibilty = 0.0
        this_internode_l = self.para.get('IBL') * pow(self.para.get('IL_AF'), self.current_age) * pow(self.para.variables_list[self.para.IL_LF], lp.layer + 1)
        terminal_flag = False
        if internode_num > 0:
            p = lp
            branch_angle = np.random.normal(self.para.variables_list[self.para.BAM], self.para.BAV) if p.forking_flag is False else np.random.normal(
                self.para.variables_list[self.para.BAM], self.para.BAV) + self.para.variables_list[self.para.FP_A]
            v_default,v_default_x=compute_dir_last_dirzx_roll_branch(self.pool[p.last_node].left_dir,self.pool[p.last_node].left_dir_x,p.rotation, branch_angle)
            v_light = v_default
            this_vlight_w = 0.0
            l_thres = this_internode_l * internode_num
            if self.global_pointcloud is not None:
                dis_mean, dir_mean, _ = compute_guide_dir(self.global_pointcloud, p.pos, v_default, l_thres)
                v_light = dir_mean
                this_vlight_w = 0.3
                if dis_mean == 100:
                    lp.type=3
                    return
                if dis_mean < l_thres:
                    this_vlight_w = 1.6
            v_tropism = self.tropism_dir
            v = v_default + v_light * this_vlight_w + self.para.variables_list[self.para.v_tropism_w] * v_tropism
            v = v / sqrt(v.dot(v.transpose()))
            for i in range(internode_num):
                if i == 0:
                    p.right_dir = v
                    p.right_l = this_internode_l
                    p.right_dir_x=v_default_x
                    p.right_rel_roll_branch = compute_rel_angles(self.pool[p.last_node].left_dir,v)
                else:
                    apical_roll_angle = np.random.rand() * 360.0
                    apical_branch_angle = np.random.normal(90, self.para.AAV)
                    last_dir=self.pool[p.last_node].left_dir if self.pool[p.last_node].right_child is None else self.pool[p.last_node].right_dir
                    p.left_dir = compute_dir(last_dir, apical_roll_angle, apical_branch_angle)
                    p.left_l=this_internode_l
                    p.left_dir_x=v_default_x
                    p.left_rel_roll_branch=compute_rel_angles(self.pool[p.last_node].left_dir,p.left_dir) if self.pool[p.last_node].right_child is None \
                            else compute_rel_angles(self.pool[p.last_node].right_dir, p.left_dir)

                if self.global_pointcloud is not None:
                    c_dis = compute_cloest_dis_from_pc(self.global_pointcloud, p.pos, p.right_dir) if i==0 \
                        else compute_cloest_dis_from_pc(self.global_pointcloud, p.pos, p.left_dir)
                    if c_dis < this_internode_l * 1.5 and self.constrain_flag:
                        terminal_flag = True
                        if c_dis < this_internode_l * 0.5:
                            break
                this_phyllotaxy = np.random.normal(self.para.variables_list[self.para.RAM],
                                                   self.para.RAV) if i == 0 else p.phyllotaxy + np.random.normal(
                    self.para.variables_list[self.para.RAM], self.para.RAV)
                this_NLB = 1
                if forking_flag:
                    rand = np.random.rand()
                    if rand < forking_possibilty:
                        this_NLB = 2
                    forking_flag = False
                for n in range(this_NLB):
                    lb = self.pool[self.nodes_counts]
                    lb.reset_node()
                    self.nodes_counts+=1
                    lb.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
                    lb.type = 3 if terminal_flag else 1
                    lb.layer = parent_level + 1
                    lb.pos = p.pos + p.right_dir * p.right_l if i==0 and n==0 else p.pos + p.left_dir * p.left_l
                    lb.created_age = self.current_age
                    lb.phyllotaxy = this_phyllotaxy
                    lb.rotation = 360.0 / this_NLB * n + lb.phyllotaxy
                    construct_graph(p, lb)
                    p = lb

                    if n != this_NLB - 1:
                        p.left_l = 0
                        p.left_dir = self.pool[p.last_node].right_dir if i==0 and n==0 else self.pool[p.last_node].left_dir
                        p.left_dir_x=v_default_x
                        p.left_rel_roll_branch = (0, 90)
            if p.id == lp.id:
                lp.type = 3
                return
            p.left_dir=self.pool[p.last_node].right_dir if p.last_node==lp.id else self.pool[p.last_node].left_dir
            p.left_l=0
            p.left_dir_x = v_default_x
            p.left_rel_roll_branch = (0, 90)
            tb = self.pool[self.nodes_counts]
            tb.reset_node()
            self.nodes_counts+=1
            tb.width=self.para.get('IBL') / self.para.variables_list[self.para.BDM]
            tb.type = 4 if terminal_flag else 0
            tb.layer = p.layer
            tb.pos = p.pos
            tb.phyllotaxy = 0
            tb.created_age = self.current_age
            construct_graph(p, tb)
            self.update_box(tb.pos)
            lp.type = 2

    def production(self):
        stack = []
        stack.append(0)
        tb_list=[]
        lb_list = []
        while len(stack) != 0:
            p_id = stack.pop()
            p=self.pool[p_id]
            if p.type==0:
                tb_list.append(p)
            if p.type==1:
                lb_list.append(p)
                #self.grow_shot_from_terminal_bud(p)
            if p.right_child is not None:
                stack.append(p.right_child)
            if p.left_child is not None:
                stack.append(p.left_child)

        for tb in tb_list:
            #print('grow_shot_from_terminal_bud(tb)')
            self.grow_shot_from_terminal_bud(tb)
        for lb in lb_list:
            #print('grow_shot_from_lateral_bud(lb)')
            self.grow_shot_from_lateral_bud(lb)
class tree_mesh:
    leaf_mean_size = 0.2
    leaf_level=2
    leaf_num = 2
    leaf_ratio = 1.0
    leaf_ry = 0
    leaf_rx = 0
    leaf_size_variance = leaf_mean_size/10
    leaf_grow_rate = 1.1
    leaf_drop_possibility = 0.1
    leaf_num_internode=4
    leaf_life_time=5
    vs=None
    vts = None
    trunk_vs = None
    leaf_vs = None
    trunk_vts = None
    leaf_vts = None
    trunk_f_layer = None
    leaf_f_layer = None
    vs_trunk_leaf_bn = None
    vts_trunk_leaf_bn = None
    trunk_f_vs = None
    trunk_f_vts = None
    leaf_f_vs = None
    leaf_f_vts = None
    billboards = None
    LOD_MAX_LAYER = None
    bark_text = None
    leaf_text = None
    billboard_angles = None
    pool=None
    ts_para=None
    def __init__(self):
        self.vs=[]
        self.vts=[(0,0),(0,1),(1,1),(1,0)]
        self.trunk_vs=[]
        self.leaf_vs=[]
        self.trunk_vts=[]
        self.leaf_vts=[(0,0),(0,1),(1,1),(1,0)]
        self.trunk_f_layer=[]
        self.leaf_f_layer=[]

        self.vs_trunk_leaf_bn=-1
        self.vts_trunk_leaf_bn=-1
        self.trunk_f_vs=[]
        self.trunk_f_vts=[]
        self.leaf_f_vs = []
        self.leaf_f_vts = []
        self.billboards=[]
        self.LOD_MAX_LAYER=1
        self.billboard_angles=1
        #self.cpr_instantce_bn=[]
    def get_max(self):
        return self.trunk_vs.max(0)
    def get_min(self):
        return self.trunk_vs.min(0)
    def get_all_vs_counts(self):
        return self.trunk_vs.shape[0]+self.leaf_vs.shape[0]
    def get_all_vs(self):
        return np.vstack((self.trunk_vs,self.leaf_vs)) if self.leaf_vs.shape[0] > 0 else self.trunk_vs
    def get_all_vts(self):
        return np.vstack((self.trunk_vts,self.leaf_vts)) if self.leaf_vts.shape[0] > 0 else self.trunk_vts
    def get_all_f_vs(self):
        return np.vstack((self.trunk_f_vs,self.leaf_f_vs)) if self.leaf_f_vs.shape[0] > 0 else self.trunk_f_vs
    def get_all_f_vts(self):
        return np.vstack((self.trunk_f_vts,self.leaf_f_vts)) if self.leaf_f_vts.shape[0] > 0 else self.trunk_f_vts
    def get_all_vts_counts(self):
        return self.trunk_vts.shape[0]+self.leaf_vts.shape[0]
    def translate(self,translation):
        #self.vs+=translation
        self.trunk_vs += translation
        if self.leaf_vs.shape[0] > 0:
            self.leaf_vs += translation
        for billboard in self.billboards:
            billboard['vs']+=translation
    def rotate(self,rotation_matrix):
        #self.vs=self.vs.dot(rotation_matrix)
        self.trunk_vs=self.trunk_vs.dot(rotation_matrix)
        self.leaf_vs = self.leaf_vs.dot(rotation_matrix)
        for billboard in self.billboards:
            billboard['vs']= billboard['vs'].dot(rotation_matrix)
    def scale(self,scalar,center):
        trunk_vs_c=self.trunk_vs-center
        trunk_vs_c*=scalar
        self.trunk_vs=trunk_vs_c+center
        if self.leaf_vs.shape[0]>0:
            leaf_vs_c = self.leaf_vs - center
            leaf_vs_c *= scalar
            self.leaf_vs = leaf_vs_c + center
        for billboard in self.billboards:
            b_vs_c=billboard['vs']-center
            billboard['vs']= b_vs_c*scalar+center

    def traverse_and_compress(self,p_id,angle_interval=45,
                              tex_height_real_length=0.5, tex_width_real_length=0.5, ty=0, TRIANGLES=True,compress_flag=True,t=12):
        p=self.pool[p_id]
        if p.id==0:
            self.traverse_and_compress(p.left_child, ty=ty, tex_width_real_length=tex_width_real_length,
                                       tex_height_real_length=tex_height_real_length, compress_flag=compress_flag,t=t)
            return
        if p.type==6:
            return
        l=self.pool[p.last_node].left_l if self.pool[p.last_node].left_child==p.id else self.pool[p.last_node].right_l
        direction = self.pool[p.last_node].left_dir if self.pool[p.last_node].left_child == p.id else self.pool[p.last_node].right_dir
        p1=self.pool[p.last_node]
        p2=p
        w1 = p1.width
        w2 = p2.width
        layer1 = p1.layer
        layer2 = p2.layer
        trunk_vs_start=-1
        leaf_vs_start=-1
        trunk_vts_start=-1
        leaf_vts_start=-1
        trunk_f_start = -1
        leaf_f_start = -1
        mark_flag=False
        if (p.last_node == None or layer2 != layer1) and layer2 <= self.LOD_MAX_LAYER:
            if l>0:
                trunk_vs_start = len(self.trunk_vs)
                trunk_vts_start = len(self.trunk_vts)
                leaf_vs_start = len(self.leaf_vs)
                trunk_f_start=len(self.trunk_f_vs)
                leaf_f_start=len(self.leaf_f_vs)
                mark_flag=True if layer2>0 else False
        if l > 0:
            if p1.layer != p2.layer:
                w1 = w2
            a_bins = round(360 / angle_interval)
            dx1 = 2 * pi * w1 / 8 / tex_width_real_length
            dx2 = 2 * pi * w2 / 8 / tex_width_real_length
            dy = l / tex_height_real_length

            if layer1 != layer2 or p1.id == 0:
                ty = 0
                roll, branch = compute_angels_from_dir(direction)
                Rotation_Matrix = compute_Rotation_Matrix(roll, branch)
                global_dir_in_circle = []
                for i in range(a_bins):
                    angle = i * angle_interval
                    angle = angle * pi / 180
                    dir = np.array([cos(angle), sin(angle), 0])
                    global_dir = Rotation_Matrix.dot(dir)
                    global_dir_in_circle.append(global_dir)
                for n in range(len(global_dir_in_circle)):
                    dir = global_dir_in_circle[n]
                    v = p1.pos + dir * w1
                    self.trunk_vs.append(v)
                    vt = ((n * dx1), ty)
                    self.trunk_vts.append(vt)
                self.trunk_vts.append((a_bins * dx1, ty))
            start_idx=len(self.trunk_vs)-a_bins
            for n in range(a_bins):
                v_1 = self.trunk_vs[start_idx + n]
                x2 = p2.pos[0]
                y2 = p2.pos[1]
                z2 = p2.pos[2]
                x1 = v_1[0]
                y1 = v_1[1]
                z1 = v_1[2]
                xd = direction[0]
                yd = direction[1]
                zd = direction[2]
                b = (x2 - x1) * xd + (y2 - y1) * yd + (z2 - z1) * zd
                yb = xd * xd + yd * yd + zd * zd
                xl = b / yb
                v_ = v_1 + direction * xl
                v_p2_base_dir = Utility.normlize_vector(v_ - p2.pos)
                v = v_p2_base_dir * w2 + p2.pos
                self.trunk_vs.append(v)
                vt = (n * dx2, ty + dy)
                self.trunk_vts.append(vt)
            self.trunk_vts.append((a_bins * dx2, ty + dy))
            vs_id_start = len(self.trunk_vs)-2*a_bins
            vts_id_start = len(self.trunk_vts)
            for n in range(a_bins):
                v_id1 = vs_id_start + 1 + n
                v_id2 = vs_id_start +a_bins+ 1 + n
                v_id3 = vs_id_start+a_bins+ + 1 + (n + 1) % a_bins
                v_id4 = vs_id_start + 1 + (n + 1) % a_bins
                vt_id1 = vts_id_start - 2 * (a_bins + 1) + 1 + n
                vt_id2 = vts_id_start - 2 * (a_bins + 1) + 1 + n + a_bins + 1
                vt_id3 = vts_id_start - 2 * (a_bins + 1) + 1 + (n + 1) + a_bins + 1
                vt_id4 = vts_id_start - 2 * (a_bins + 1) + 1 + (n + 1)
                if TRIANGLES:
                    f1 = (v_id1, v_id2, v_id4)
                    f_t1 = (vt_id1, vt_id2, vt_id4)
                    self.trunk_f_vs.append(f1)
                    self.trunk_f_vts.append(f_t1)
                    self.trunk_f_layer.append(layer2)

                    f2 = (v_id2, v_id3, v_id4)
                    f_t2 = (vt_id2, vt_id3, vt_id4)
                    self.trunk_f_vs.append(f2)
                    self.trunk_f_vts.append(f_t2)
                    self.trunk_f_layer.append(layer2)
                else:
                    f = (v_id1, v_id2, v_id3, v_id4)
                    f_t = (vt_id1, vt_id2, vt_id3, vt_id4)
                    self.trunk_f_vs.append(f)
                    self.trunk_f_vts.append(f_t)
                    self.trunk_f_layer.append(layer2)
            ty += dy
        #terminal ball
        if p.left_child is None or self.pool[p.left_child].type==6:
            last_p = p
            while last_p.last_node is not None and self.pool[last_p.last_node].left_l == 0:
                last_p = self.pool[last_p.last_node]
            lp2=last_p
            w2 = lp2.width
            roll, branch = compute_angels_from_dir(direction)
            Rotation_Matrix = compute_Rotation_Matrix(roll, branch)
            global_dir_in_circle = []
            a_bins = 8
            h_bins = 3
            angle_interval = 45
            for i in range(a_bins):
                angle = i * angle_interval
                angle = angle * pi / 180
                dir = np.array([cos(angle), sin(angle), 0])
                global_dir = Rotation_Matrix.dot(dir)
                global_dir_in_circle.append(global_dir)
            ratio = 2 * w2 / tex_height_real_length
            dy = 1 / 6 * ratio
            dx = 1 / 8 * ratio
            for h in range(h_bins + 1):
                h_angle = 30 * h
                w = w2 * cos(h_angle * pi / 180)
                for n in range(len(global_dir_in_circle)):
                    dir = global_dir_in_circle[n]
                    v = p2.pos + direction * w2 * sin(h_angle * pi / 180) + dir * w
                    self.trunk_vs.append(v)
                    vt = (n * dx, h * dy)
                    self.trunk_vts.append(vt)
                self.trunk_vts.append((a_bins * dx, h * dy))
                if h > 0:
                    vs_id_start=len(self.trunk_vs)
                    vts_id_start=len(self.trunk_vts)
                    for n in range(a_bins):
                        v_id1 = vs_id_start - 2 * a_bins + 1 + n
                        v_id2 = vs_id_start - 2 * a_bins + 1 + n + a_bins
                        v_id3 = vs_id_start - 2 * a_bins + 1 + (n + 1) % a_bins + a_bins
                        v_id4 = vs_id_start - 2 * a_bins + 1 + (n + 1) % a_bins
                        vt_id1 = vts_id_start - 2 * (a_bins + 1) + 1 + n
                        vt_id2 = vts_id_start - 2 * (a_bins + 1) + 1 + n + a_bins + 1
                        vt_id3 = vts_id_start - 2 * (a_bins + 1) + 1 + (n + 1) + a_bins + 1
                        vt_id4 = vts_id_start - 2 * (a_bins + 1) + 1 + (n + 1)

                        if TRIANGLES:
                            f1 = (v_id1, v_id2, v_id4)
                            f_t1 = (vt_id1, vt_id2, vt_id4)
                            self.trunk_f_vs.append(f1)
                            self.trunk_f_vts.append(f_t1)
                            self.trunk_f_layer.append(layer2)

                            f2 = (v_id2, v_id3, v_id4)
                            f_t2 = (vt_id2, vt_id3, vt_id4)
                            self.trunk_f_vs.append(f2)
                            self.trunk_f_vts.append(f_t2)
                            self.trunk_f_layer.append(layer2)
                        else:
                            f = (v_id1, v_id2, v_id3, v_id4)
                            f_t = (vt_id1, vt_id2, vt_id3, vt_id4)
                            self.trunk_f_vs.append(f)
                            self.trunk_f_vts.append(f_t)
                            self.trunk_f_layer.append(layer2)
        if l > 0:
            leaf_counts = 0
            roll, branch = compute_angels_from_dir(direction)
            Rotation_Matrix = compute_Rotation_Matrix(roll, branch)
            node = p2
            leaf_year_old = max(self.ts_para.get('t') -1- node.created_age,0)
            basic_width=(self.ts_para.get('IBL') / self.ts_para.variables_list[self.ts_para.BDM]*1.001)
            rotation=360/self.leaf_num_internode
            if leaf_year_old <= self.leaf_life_time and p2.width<basic_width:
                leaf_size = self.leaf_mean_size * pow(self.leaf_grow_rate, leaf_year_old)
                leaf_size = np.random.normal(leaf_size, self.leaf_size_variance)*self.ts_para.get('IBL')
                for nn in range(self.leaf_num_internode):
                    for ln in range(self.leaf_num):
                        rand = np.random.rand()
                        drop_possibility=1-pow(1-self.leaf_drop_possibility,leaf_year_old)
                        if rand < drop_possibility:
                            continue
                        this_dir_z=direction
                        this_dir_x=np.cross(direction,np.array([0,0,1]))
                        this_dir_y=np.cross(this_dir_z,this_dir_x)
                        this_Rotation_Matrix=np.array([this_dir_x,this_dir_y,this_dir_z]).transpose()
                        this_Rotation_Matrix=(this_Rotation_Matrix).dot(Rotation_Matrix_from_z_rotation(nn*rotation+ln*360/(self.leaf_num)))
                        this_Rotation_Matrix=this_Rotation_Matrix.dot(Rotation_Matrix_from_x_rotation(np.random.normal(0,20)))
                        this_Rotation_Matrix = this_Rotation_Matrix.dot(
                            Rotation_Matrix_from_y_rotation(np.random.normal(0, 20)))
                        dir_x=this_Rotation_Matrix[:,2] if abs(direction[2])<0.707 else this_Rotation_Matrix[:,1]
                        dir_z = this_Rotation_Matrix[:,0]
                        z_roll, z_branch = compute_angels_from_dir(dir_z)
                        z_Rotation_Matrix = compute_Rotation_Matrix(z_roll, max(z_branch-np.random.normal(45, 5),-90))
                        dir_z=z_Rotation_Matrix[:,2]
                        #leaf_pos=p2.pos+dir_z*w2
                        leaf_pos = p1.pos +direction*l*(nn+1)/self.leaf_num_internode+dir_z * w2
                        leaf_v1 = leaf_pos - dir_x * leaf_size / 2
                        leaf_v2 = leaf_pos - dir_x * leaf_size / 2 + dir_z * leaf_size * self.leaf_ratio
                        leaf_v3 = leaf_pos + dir_x * leaf_size / 2 + dir_z * leaf_size * self.leaf_ratio
                        leaf_v4 = leaf_pos + dir_x * leaf_size / 2
                        self.leaf_vs.append(leaf_v1)
                        self.leaf_vs.append(leaf_v2)
                        self.leaf_vs.append(leaf_v3)
                        self.leaf_vs.append(leaf_v4)
                        vs_id_start=len(self.leaf_vs)
                        v_id1 = vs_id_start - 4 + 1
                        v_id2 = vs_id_start - 4 + 2
                        v_id3 = vs_id_start - 4 + 3
                        v_id4 = vs_id_start - 4 + 4
                        #leaf_counts+=1
                        if TRIANGLES:
                            f1 = (v_id1, v_id2, v_id4)
                            f_t1 = (1, 2, 4)
                            self.leaf_f_vs.append(f1)
                            self.leaf_f_vts.append(f_t1)
                            self.leaf_f_layer.append(layer2)

                            f2 = (v_id2, v_id3, v_id4)
                            f_t2 = (2, 3, 4)
                            self.leaf_f_vs.append(f2)
                            self.leaf_f_vts.append(f_t2)
                            self.leaf_f_layer.append(layer2)
                        else:
                            f = (v_id1, v_id2, v_id3, v_id4)
                            f_t = (1, 2, 3, 4)
                            self.leaf_f_vs.append(f)
                            self.leaf_f_vts.append(f_t)
                            self.leaf_f_layer.append(layer2)
        if p.left_child is not None:
            self.traverse_and_compress(p.left_child, ty=ty, tex_width_real_length=tex_width_real_length,
                                       tex_height_real_length=tex_height_real_length, compress_flag=compress_flag,t=t)
        if p.right_child is not None:
            self.traverse_and_compress(p.right_child, ty=ty, tex_width_real_length=tex_width_real_length,
                                       tex_height_real_length=tex_height_real_length, compress_flag=compress_flag,t=t)
        if mark_flag and compress_flag:
            '''compress'''
            start=time.time()
            cpr_trunk_vs=np.array(self.trunk_vs[trunk_vs_start:])
            cpr_leaf_vs=np.array(self.leaf_vs[leaf_vs_start:])
            cpr_vs=np.vstack((cpr_trunk_vs,cpr_leaf_vs)) if cpr_leaf_vs.shape[0]>0 else cpr_trunk_vs
            cpr_vts_1 = [(0,0),(0,1),(1,1),(1,0)]
            cpr_vts_2=self.trunk_vts[trunk_vts_start:].copy()
            cpr_vts=np.zeros([4+len(cpr_vts_2),2])
            cpr_vts[:4]=cpr_vts_1
            cpr_vts[4:]=cpr_vts_2
            cpr_trunk_f_vs = np.array(self.trunk_f_vs[trunk_f_start:]) - trunk_vs_start - 1
            cpr_trunk_f_vts = np.array(self.trunk_f_vts[trunk_f_start:]) - trunk_vts_start + 4 - 1
            cpr_leaf_f_vs = np.array(self.leaf_f_vs[leaf_f_start:]) - leaf_vs_start - 1+cpr_trunk_vs.shape[0]
            cpr_leaf_f_vts = np.array(self.leaf_f_vts[leaf_f_start:]) - 1
            cpr_f_vs = np.vstack((cpr_leaf_f_vs,cpr_trunk_f_vs)) if cpr_leaf_f_vs.shape[0] > 0 else cpr_trunk_f_vs
            cpr_f_vts = np.vstack((cpr_leaf_f_vts,cpr_trunk_f_vts)) if cpr_leaf_f_vts.shape[0] > 0 else cpr_trunk_f_vts
            bd = cpr_leaf_f_vs.shape[0]
            anchor_point = np.array(p1.pos)
            eval,evec=Utility.PCA(cpr_trunk_vs)
            V_from_p2=cpr_trunk_vs-p2.pos
            mean_V=Utility.normlize_vector(V_from_p2.mean(0))

            for nn in range(self.billboard_angles):
                R=np.eye(3,dtype=float)
                if nn==0:
                    #R=evec
                    R[:, 2] = mean_V
                    temp_r0=np.cross(mean_V, np.array([0, 0, 1], dtype=float))
                    if sqrt(temp_r0.dot(temp_r0.transpose()))<0.01:
                        R[:,0]=np.array([1,0,0],dtype=float)
                        R[:, 1] = Utility.normlize_vector(np.cross(mean_V, R[:, 0]))
                    else:
                        R[:, 0] = Utility.normlize_vector(temp_r0)
                        R[:, 1] = Utility.normlize_vector(np.cross(mean_V, R[:, 0]))
                if nn==1:
                    R[:, 2] = mean_V
                    temp_r0 = np.cross(mean_V, np.array([0, 0, 1], dtype=float))
                    if sqrt(temp_r0.dot(temp_r0.transpose())) < 0.01:
                        R[:, 1] = np.array([1, 0, 0], dtype=float)
                        R[:, 0] = Utility.normlize_vector(np.cross(mean_V, R[:, 1]))
                    else:
                        R[:, 1] = Utility.normlize_vector(temp_r0)
                        R[:, 0] = Utility.normlize_vector(np.cross(mean_V, R[:, 1]))
                resolution=round(20000/(pow(2,layer2-1)))
                max_height=round(1000/(pow(2,layer2-1)))
                image, four_corners = CUDA_print_image(cpr_vs, cpr_vts, cpr_f_vs, cpr_f_vts, R, self.leaf_text,
                                                       self.bark_text, bd, anchor_point, resolution=resolution,max_height=max_height)
                billboard_dir = {}
                billboard_dir['vs'] = four_corners.astype(np.float32)
                billboard_dir['vts'] = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], np.float32)
                billboard_dir['f_vs'] = np.array([[1, 2, 3], [1, 3, 4]], np.uint32)
                billboard_dir['f_vts'] = np.array([[1, 2, 3], [1, 3, 4]], np.uint32)
                billboard_dir['image'] = image
                billboard_dir['layer'] = layer2
                #billboard_dir['id'] = inter.id
                billboard_dir['angle'] = nn
                billboard_dir['front'] = R[:, 1] * -1.
                end = time.time()
                time_spending = end - start
                billboard_dir['time']=time_spending
                self.billboards.append(billboard_dir)

    def from_lpy(self,ts:tree_scene,leaf_text=None, bark_text=None, tex_width_real_length=0.5,
                 tex_height_real_length=0.5,compress_flag=False):
        self.pool=ts.pool.copy()
        first_node=self.pool[0]
        self.bark_text=bark_text
        self.leaf_text=leaf_text
        self.ts_para=ts.para
        t=round(ts.para.variables_list[ts.para.t])
        self.traverse_and_compress(first_node.id, tex_height_real_length=tex_height_real_length,
                                   tex_width_real_length=tex_width_real_length,compress_flag=compress_flag,t=t)
        self.trunk_vs = np.array(self.trunk_vs)
        self.trunk_vts = np.array(self.trunk_vts)
        self.leaf_vs = np.array(self.leaf_vs)
        self.leaf_vts = np.array(self.leaf_vts)
        self.trunk_f_vs = np.array(self.trunk_f_vs,dtype=int)
        self.trunk_f_vts = np.array(self.trunk_f_vts,dtype=int)
        self.leaf_f_vs = np.array(self.leaf_f_vs,dtype=int)
        self.leaf_f_vts = np.array(self.leaf_f_vts,dtype=int)
        self.trunk_f_layer = np.array(self.trunk_f_layer)
        self.leaf_f_layer = np.array(self.leaf_f_layer)

    def save_obj(self,filename,leaf_filename,bark_filename,mtl_filename=None,leaf_flag=True):

        mtl_filename = filename[:-3] + 'mtl' if mtl_filename is None else mtl_filename
        f = open(filename, 'w+')
        line = '# generated by WX with python-3.9 in'
        import datetime
        line += datetime.datetime.today().ctime()
        f.write(line + '\n')
        # write vertex
        for i in range(self.trunk_vs.shape[0]):
            x_str = str(round(self.trunk_vs[i, 0], 3))
            y_str = str(round(self.trunk_vs[i, 1], 3))
            z_str = str(round(self.trunk_vs[i, 2], 3))
            line = 'v ' + x_str + ' ' + y_str + ' ' + z_str + '\n'
            f.write(line)
        for i in range(self.leaf_vs.shape[0]):
            x_str = str(round(self.leaf_vs[i, 0], 3))
            y_str = str(round(self.leaf_vs[i, 1], 3))
            z_str = str(round(self.leaf_vs[i, 2], 3))
            line = 'v ' + x_str + ' ' + y_str + ' ' + z_str + '\n'
            f.write(line)
        for i in range(self.trunk_vts.shape[0]):
            x_str = str(round(self.trunk_vts[i, 0], 3))
            y_str = str(round(self.trunk_vts[i, 1], 3))
            line = 'vt ' + x_str + ' ' + y_str + '\n'
            f.write(line)
        for i in range(self.leaf_vts.shape[0]):
            x_str = str(round(self.leaf_vts[i, 0], 3))
            y_str = str(round(self.leaf_vts[i, 1], 3))
            line = 'vt ' + x_str + ' ' + y_str + '\n'
            f.write(line)
        f.write('\n')
        # write face
        # truck
        mtl = mtl_filename.split('/')[-1]

        f.write('mtllib ' + mtl + '\n')
        f.write('usemtl Color_0\n')
        f.write('o trucks\n')
        for i in range(self.trunk_f_vs.shape[0]):
            line = 'f'
            for n in range(self.trunk_f_vs.shape[1]):
                v_id_str = str(self.trunk_f_vs[i][n])
                vt_id_str = str(self.trunk_f_vts[i][n])
                line += ' ' + v_id_str + '/' + vt_id_str
            f.write(line + '\n')

        # leaf
        if leaf_flag:
            f.write('usemtl Color_1\n')
            f.write('o leafs\n')
            leaf_vs_id_start=self.trunk_vs.shape[0]
            leaf_vts_id_start=self.trunk_vts.shape[0]
            for i in range(self.leaf_f_vs.shape[0]):
                # f.write('usemtl Color_1\n')
                # f.write('o leaf'+str(i)+'\n')
                line = 'f'
                for n in range(self.leaf_f_vs.shape[1]):
                    v_id_str = str(self.leaf_f_vs[i][n]+leaf_vs_id_start)
                    vt_id_str = str(self.leaf_f_vts[i][n]+leaf_vts_id_start)
                    line += ' ' + v_id_str + '/' + vt_id_str
                f.write(line + '\n')
        f.close()

        start = filename[:filename.rfind('/')]
        leaf = os.path.relpath(leaf_filename, start)
        bark = os.path.relpath(bark_filename, start)
        leaf_filename=leaf
        bark_filename=bark
        # leaf = leaf_filename.split('/')[-1]
        # bark = bark_filename.split('/')[-1]
        f = open(mtl_filename, 'w+')

        f.write('newmtl Color_1\n')
        f.write('	Ka 0.2 0.2 0.2\n')
        f.write('	Kd 1 1 1\n')
        f.write('	Ks 0.2 0.2 0.2\n')
        f.write('	Tr 0\n')
        f.write('	illum 1\n')
        f.write('	Ns 50\n')
        f.write('map_Kd ' + leaf_filename + '\n')

        f.write('newmtl Color_0\n')
        f.write('	Ka 0.2 0.2 0.2\n')
        f.write('	Kd 1 1 1\n')
        f.write('	Ks 0.2 0.2 0.2\n')
        f.write('	Tr 0\n')
        f.write('	illum 1\n')
        f.write('	Ns 50\n')
        f.write('map_Kd ' + bark_filename)
        f.close()
    def save_lod(self,out_path,leaf_filename,bark_filename,angle=-1):
        Utility.creat_path(out_path)
        time_filename = out_path + '/time.txt'
        time_file = open(time_filename, 'w')
        for i in range(1,self.LOD_MAX_LAYER+1):
            filename=out_path+'/lod'+str(i)+'.obj'
            mtl_filename = filename[:-3] + 'mtl'
            print(filename,mtl_filename)
            file = open(filename, 'w')
            line = '# generated by WX with python-3.9 in'
            import datetime
            line += datetime.datetime.today().ctime()
            file.write(line + '\n')
            # write vertex
            new_index = {}
            new_index_vt = {}
            new_trunk_vs = []
            new_trunk_vts = []
            new_trunk_f_vs = []
            new_trunk_f_vts = []
            for n in range(self.trunk_f_layer.shape[0]):
                if self.trunk_f_layer[n]<i:
                    f=self.trunk_f_vs[n]
                    f_t=self.trunk_f_vts[n]
                    f_vs=np.zeros(3)
                    f_vts=np.zeros(3)
                    for m in range(3):
                        fv=f[m]-1
                        fvt=f_t[m]-1
                        if fv in new_index:
                            new_fv=new_index[fv]
                        else:
                            new_fv=len(new_trunk_vs)
                            new_index[fv]=new_fv
                            new_trunk_vs.append(self.trunk_vs[fv])

                        if fvt in new_index_vt:
                            new_fvt=new_index_vt[fvt]
                        else:
                            new_fvt = len(new_trunk_vts)
                            new_index_vt[fvt] = new_fvt
                            new_trunk_vts.append(self.trunk_vts[fvt])
                        f_vs[m]=new_fv+1
                        f_vts[m]=new_fvt+1
                    new_trunk_f_vs.append(f_vs)
                    new_trunk_f_vts.append(f_vts)

            new_index = {}
            new_index_vt = {}
            new_leaf_vs = []
            new_leaf_vts = []
            new_leaf_f_vs = []
            new_leaf_f_vts = []
            for n in range(self.leaf_f_layer.shape[0]):
                if self.leaf_f_layer[n] < i:
                    f = self.leaf_f_vs[n]
                    f_t = self.leaf_f_vts[n]
                    f_vs = np.zeros(3)
                    f_vts = np.zeros(3)
                    for m in range(3):
                        fv = f[m]-1
                        fvt = f_t[m]-1
                        if fv in new_index:
                            new_fv = new_index[fv]
                        else:
                            new_fv = len(new_leaf_vs)
                            new_index[fv] = new_fv
                            new_leaf_vs.append(self.leaf_vs[fv])

                        if fvt in new_index_vt:
                            new_fvt = new_index_vt[fvt]
                        else:
                            new_fvt = len(new_leaf_vts)
                            new_index_vt[fvt] = new_fvt
                            new_leaf_vts.append(self.leaf_vts[fvt])
                        f_vs[m] = new_fv+1
                        f_vts[m] = new_fvt+1
                    new_leaf_f_vs.append(f_vs)
                    new_leaf_f_vts.append(f_vts)


            for n in range(len(new_trunk_vs)):
                x_str = str(round(new_trunk_vs[n][0], 3))
                y_str = str(round(new_trunk_vs[n][1], 3))
                z_str = str(round(new_trunk_vs[n][2], 3))
                line = 'v ' + x_str + ' ' + y_str + ' ' + z_str + '\n'
                file.write(line)
            for n in range(len(new_leaf_vs)):
                x_str = str(round(new_leaf_vs[n][0], 3))
                y_str = str(round(new_leaf_vs[n][1], 3))
                z_str = str(round(new_leaf_vs[n][2], 3))
                line = 'v ' + x_str + ' ' + y_str + ' ' + z_str + '\n'
                file.write(line)
            for n in range(len(new_trunk_vts)):
                x_str = str(round(new_trunk_vts[n][0], 3))
                y_str = str(round(new_trunk_vts[n][1], 3))
                line = 'vt ' + x_str + ' ' + y_str + '\n'
                file.write(line)
            for n in range(len(new_leaf_vts)):
                x_str = str(round(new_leaf_vts[n][0], 3))
                y_str = str(round(new_leaf_vts[n][1], 3))
                line = 'vt ' + x_str + ' ' + y_str + '\n'
                file.write(line)
            file.write('\n')
            # write face
            # truck
            mtl = mtl_filename.split('/')[-1]
            file.write('mtllib ' + mtl + '\n')
            file.write('usemtl Color_0\n')
            file.write('o trucks\n')
            for m in range(len(new_trunk_f_vs)):
                line = 'f'
                for n in range(3):
                    v_id_str = str(int(new_trunk_f_vs[m][n]))
                    vt_id_str = str(int(new_trunk_f_vts[m][n]))
                    line += ' ' + v_id_str + '/' + vt_id_str
                file.write(line + '\n')

            # leaf
            if len(new_leaf_f_vs)>0:
                file.write('usemtl Color_1\n')
                file.write('o leafs\n')
                leaf_vs_id_start=len(new_trunk_vs)
                leaf_vts_id_start=len(new_trunk_vts)
                for m in range(len(new_leaf_f_vs)):
                    line = 'f'
                    for n in range(3):
                        v_id_str = str(int(new_leaf_f_vs[m][n]+leaf_vs_id_start))
                        vt_id_str = str(int(new_leaf_f_vts[m][n]+leaf_vts_id_start))
                        line += ' ' + v_id_str + '/' + vt_id_str
                    file.write(line + '\n')
            counts=0
            width_counts=0
            texture_path=out_path+'/texture'+str(i)
            Utility.creat_path(texture_path)
            billboard_all_vs=[]
            billboard_all_f_vs=[]
            times = 0
            for billboard in self.billboards:
                if billboard['layer']==i and (billboard['angle']==angle or angle==-1):
                    vs=billboard['vs']
                    vts=billboard['vts']
                    f_vs=billboard['f_vs']
                    f_vts=billboard['f_vts']
                    image=billboard['image']
                    times+=billboard['time']
                    for n in range(vs.shape[0]):
                        x_str = str(round(vs[n, 0], 3))
                        y_str = str(round(vs[n, 1], 3))
                        z_str = str(round(vs[n, 2], 3))
                        if angle == -1:
                            billboard_all_vs.append([vs[n,0],vs[n,1],vs[n,2]])
                        line = 'v ' + x_str + ' ' + y_str + ' ' + z_str + '\n'
                        file.write(line)
                    for n in range(vts.shape[0]):
                        x_str = str(round(vts[n, 0], 3))
                        y_str = str(round(vts[n, 1], 3))
                        line = 'vt ' + x_str + ' ' + y_str + '\n'
                        file.write(line)
                    file.write('usemtl Color_'+str(2+counts)+'\n')
                    file.write('o billborad'+str(counts)+'\n')
                    vs_id_start = len(new_trunk_vs) + len(new_leaf_vs)+4*counts
                    vts_id_start = len(new_trunk_vts)+len(new_leaf_vts)+4*counts
                    for m in range(f_vs.shape[0]):
                        line = 'f'
                        for n in range(3):
                            v_id_str = str(f_vs[m][n] + vs_id_start)
                            vt_id_str = str(f_vts[m][n] + vts_id_start)
                            line += ' ' + v_id_str + '/' + vt_id_str
                        if angle == -1:
                            billboard_all_f_vs.append([f_vs[m][0] + 4 * counts,f_vs[m][1] + 4 * counts,f_vs[m][2] + 4 * counts])
                        file.write(line + '\n')

                    cv2.imwrite(texture_path+'/'+str(counts)+'.png',image)
                    counts += 1
            time_file.write(f'{times}\n')
            file.close()
            if angle==-1:
                Utility.save_general_obj(out_path+'/billboards_lod'+str(i)+'.obj',np.array(billboard_all_vs),np.array(billboard_all_f_vs))
            start = filename[:filename.rfind('/')]
            leaf = os.path.relpath(leaf_filename, start)
            bark = os.path.relpath(bark_filename, start)
            #leaf_filename=leaf
            #bark_filename=bark

            texture_path=os.path.relpath(texture_path,start)
            #bill=os.path.relpath(big_image_filename, start)
            f = open(mtl_filename, 'w')

            f.write('newmtl Color_1\n')
            f.write('	Ka 0.2 0.2 0.2\n')
            f.write('	Kd 1 1 1\n')
            f.write('	Ks 0.2 0.2 0.2\n')
            f.write('	Tr 0\n')
            f.write('	illum 1\n')
            f.write('	Ns 50\n')
            f.write('map_Kd ' + leaf + '\n')

            f.write('newmtl Color_0\n')
            f.write('	Ka 0.2 0.2 0.2\n')
            f.write('	Kd 1 1 1\n')
            f.write('	Ks 0.2 0.2 0.2\n')
            f.write('	Tr 0\n')
            f.write('	illum 1\n')
            f.write('	Ns 50\n')
            f.write('map_Kd ' + bark+'\n')
            for n in range(counts):
                f.write('newmtl Color_'+str(2+n)+'\n')
                f.write('	Ka 0.2 0.2 0.2\n')
                f.write('	Kd 1 1 1\n')
                f.write('	Ks 0.2 0.2 0.2\n')
                f.write('	Tr 0\n')
                f.write('	illum 1\n')
                f.write('	Ns 50\n')
                #bill = os.path.relpath(texture_path+'/'+str(n)+'.png', start)
                bill = texture_path + '/' + str(n) + '.png'
                f.write('map_Kd ' + bill+'\n')
            f.close()
        time_file.close()
    def get_max_bound(self):
        return self.trunk_vs.max(axis=0)
    def get_min_bound(self):
        return self.trunk_vs.min(axis=0)

def derive(global_pool,para,revise_flag=True):
    ts=tree_scene(para,global_pool)
    ts.Start()
    derivation_length = round(ts.para.variables_list[para.t])
    for i in range(derivation_length):
        #start=time.time()
        flag = ts.StartEach()
        if flag is False:
            break
        #t_start=time.time()
        ts.production()
        #t_end=time.time()
        #print('production',t_end-t_start)
        ts.EndEach()
        #end=time.time()
        #print('year', i,end-start)

    ts.End(revise_flag)
    return ts
def derive_with_gpc(global_pool,pointcloud_array,h,para:config.parameter,revise_flag=True):
   # print('%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    height = h
    scaled_pointcloud = pointcloud_array * height
    min_h=scaled_pointcloud.min(0)[2]
    ts2=tree_scene(para,global_pool)
    ts2.global_pointcloud=scaled_pointcloud
    ts2.global_height=height
    ts2.global_mid_height=(height+min_h)/2.
    ts2.Start()
    ts2.para.set('t',round(ts2.para.variables_list[ts2.para.t])+4)
    derivation_length = round(ts2.para.variables_list[ts2.para.t])

    jump_flag=False
    for i in range(derivation_length):
        #print('year',i)
        flag = ts2.StartEach()
        if flag is False:
            jump_flag=True
            break
        ts2.production()
        ts2.EndEach()
    if jump_flag is False:
        ts2.bending()
    ts2.End(revise_flag)
    height2 = ts2.box.max_z
    scaled_pointcloud2 = pointcloud_array * height2
    dis3, dis4, terminals_pos = ts2.compute_dis(scaled_pointcloud2)
    print(dis4,height2)
    dis3 /= height2
    dis4 /= height2
    ts2.para.set('t', round(ts2.para.variables_list[ts2.para.t]) - 4)
    #print('dis3: ', dis3, 'dis4: ', dis4)
    return dis3, dis4, ts2

