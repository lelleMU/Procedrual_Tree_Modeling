import os.path

import numpy as np
import Utility
import config
import lpy
import global_node_pool
import cv2
from math import *
import argparse

temp_para=config.parameter()
variables_num=len(temp_para.variables_list)
sample_num=round(np.prod(temp_para.variables_limits[:,2]))

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_PATH', default='data/example1', help='Make sure the source data path, which includes '
                                                         'the point cloud data and texture images')
parser.add_argument('--OUT_PATH',default='out/example1',help='Make sure the out path')

parser.add_argument('--GROUND_H',type=float,default=-999,help='The height of the ground')
parser.add_argument('--MAX_ITERATION',type=int,default=100,help='Maximal iterations number')
parser.add_argument('--CONFIG_FILENAME',default='xxx.txt',help='parameters configure filename')

FLAGS = parser.parse_args()
data_path=FLAGS.DATA_PATH
out_path=FLAGS.OUT_PATH
Ground_h=FLAGS.GROUND_H
max_iter=FLAGS.MAX_ITERATION
config_filename=FLAGS.CONFIG_FILENAME
Utility.creat_path(out_path)
if Ground_h == -999:
    points = np.loadtxt(f'{data_path}/data.txt')
    Ground_h = np.min(points[:, 2]) - 3
global_pool = []
for i in range(lpy.MAX_NODES_SIZE * 8):
    '''MAX_NODES_SIZE should be adjusted according to the memory size '''
    node_i = global_node_pool.Mynode()
    node_i.id = i
    global_pool.append(node_i)

def from_para_to_tree_mesh(global_pool,gpc,para:config.parameter,this_sample_num=5):
    dis4=999.
    temp=lpy.derive(global_pool,para)
    h=temp.box.max_z
    #dis4_all=0
    best_ts=None
    dis_all=0.
    for i in range(this_sample_num):
        _, t_dis4,ts= lpy.derive_with_gpc(global_pool,gpc,h,para)
        if this_sample_num>1:
            print(i,this_sample_num,t_dis4)
        dis_all+=t_dis4
        if (t_dis4)<(dis4):
            dis4 = t_dis4
            best_ts=ts
    return 0,dis_all/this_sample_num,best_ts
def load_gpc(pointcloud_filename,floor_h):
    gpc_array=Utility.read_txt_pointcloud(pointcloud_filename)
    max_ = gpc_array.max(axis=0)
    max_z=max_[2]
    h = max_z - floor_h
    center=gpc_array.mean(axis=0)
    gpc_trans=gpc_array+np.array([-center[0],-center[1],-floor_h])
    gpc_trans=gpc_trans*(1/h)
    return gpc_trans,center,h
def random_sample():
    para=config.parameter()
    for nn in range(variables_num):
        min = para.variables_limits[nn, 0]
        max = para.variables_limits[nn, 1]
        bins = para.variables_limits[nn, 2]
        if bins==1:
            para.variables_list[nn]=min
        else:
            random=np.random.rand()
            para.variables_list[nn]=min+(max-min)*random
    return para

def normal_sample(para_mean,para_std):
    para=config.parameter()
    for nn in range(variables_num):
        min = para.variables_limits[nn, 0]
        max = para.variables_limits[nn, 1]
        bins = para.variables_limits[nn, 2]
        if bins==1:
            para.variables_list[nn]=min
        else:
            mean=(para_mean[nn]-min)/(max-min)
            std=para_std[nn]
            random=np.clip(np.random.normal(mean,std),0,1)
            para.variables_list[nn]=min+(max-min)*random
    return para

def from_para_to_tree_mesh_without_guide(global_pool,gpc,para:config.parameter,revise_flag=True):
    ts=lpy.derive(global_pool,para,revise_flag)
    height2 = ts.box.max_z
    scaled_pointcloud2 = gpc * height2
    dis3, dis4, terminals_pos = ts.compute_dis(scaled_pointcloud2)
    dis3 /= height2
    dis4 /= height2
    return 0,dis4,ts

def Control_Para_Aanlysis(gpc_array):
    M = 5
    mean_para = config.parameter()
    mean_para.variables_list *= 0
    para_num = mean_para.variables_list.shape[0]
    mean_para_std = np.zeros(para_num)
    #para_config_filename = f'{out_path}/para_config.txt'
    #file = open(para_config_filename, 'w')
    this_wp = config.parameter()
    for n in range(para_num):
        print(f'processing:{temp_para.paramters_dir[n]}')
        bottom = temp_para.variables_limits[n][0]
        top = temp_para.variables_limits[n][1]
        bins = temp_para.variables_limits[n][2]
        if bins == 1:
            #file.write(f'{temp_para.paramters_dir[n]} {bottom} 0\n')
            mean_para.variables_list[n] += bottom
            mean_para_std[n] += 0
            this_wp.variables_list[n] = bottom
        else:
            all_dis4 = []
            all_val = []
            for m in range(M):
                para = random_sample()
                para.variables_list[n] = bottom
                _,dis4_b,tsb=from_para_to_tree_mesh_without_guide(global_pool,gpc_array,para)
                para.variables_list[n] = top

                _, dis4_t, tst=from_para_to_tree_mesh_without_guide(global_pool,gpc_array,para)
                print(f'processing:{temp_para.paramters_dir[n]} {m}/{M}')
                print(f'{dis4_b, dis4_t}')

                if dis4_t < 1.0 and dis4_b < 1.0:
                    all_dis4.append(dis4_t)
                    all_val.append(1)
                    all_dis4.append(dis4_b)
                    all_val.append(0)

            all_dis4 = np.array(all_dis4)
            all_val = np.array(all_val)
            mean_dis4_top = all_dis4[:int(all_dis4.shape[0] / 2)].mean()
            mean_dis4_bottom = all_dis4[int(all_dis4.shape[0] / 2):].mean()

            NN = all_dis4.shape[0]
            top_N = round(NN * 0.5)
            top_idx = all_dis4.argsort()[:top_N]
            all_dis4 = all_dis4[top_idx]
            all_val = all_val[top_idx]

            weight = np.exp(all_dis4 * -10)
            this_para_mean = (all_val * weight).sum() / weight.sum()
            add = 0
            for m in range(all_dis4.shape[0]):
                temp = all_val[m] - this_para_mean
                add += (temp * temp) * weight[m]
            this_para_std = sqrt(add / weight.sum())
            # file.write(
            #     f'{temp_para.paramters_dir[n]} {bottom + this_para_mean * (top - bottom)} {this_para_std} {mean_dis4_bottom} {mean_dis4_top}\n')
            mean_para.variables_list[n] += (bottom + this_para_mean * (top - bottom))
            mean_para_std[n] += this_para_std
            this_wp.variables_list[n] = bottom + this_para_mean * (top - bottom)
    #file.close()
    return mean_para.variables_list,mean_para_std
    # _, dis4, _ = from_para_to_tree_mesh_dis4(global_pool, gpc_array, 0,
    #                                          para=this_wp,
    #                                          this_sample_num=1)
    #file.write(f'wp dis4:{dis4}\n')
    #mean_dis4.append(dis4)

def Metropolis_Hastings_optimization(gpc_array,para_mean,para_std):
    log_path = f'{out_path}/log'
    Utility.creat_path(log_path)
    process = np.zeros(max_iter)

    t_para = config.parameter()
    batch_id = 0
    max_temperature=4.0
    while batch_id < max_iter:
        print(f'optimization:  {batch_id}/{max_iter}')
        temperature = max_temperature * (max_iter - batch_id) / max_iter
        std_ratio = 2 - (batch_id / max_iter) * (2 - 0.5)
        random_para = normal_sample(para_mean, para_std * std_ratio)
        _, dis4, ts = from_para_to_tree_mesh_without_guide(global_pool, gpc_array, random_para)
        flag = False
        if batch_id == 0:
            process[0] = dis4
            t_para = random_para
            flag = True
        else:
            temp_dis34 = process[batch_id - 1] * 1000
            this_dis34 = dis4 * 1000
            alpha = pow(exp(-this_dis34) / exp(-temp_dis34), 1 / temperature)
            print(f'termp:{temperature},trans_possibilty:{alpha}')
            rand = np.random.rand()
            if rand < alpha:
                # transmission
                process[batch_id] = dis4
                t_para = random_para
                flag = True
            else:
                process[batch_id] = process[batch_id - 1]
        if flag:
            print(f'trans...{process[batch_id]}')
            # break
        else:
            # decline_times += 1
            print(f'decline..{process[batch_id]}')
        batch_id += 1
    log_filename = f'{log_path}/LOG.txt'
    log_file = open(log_filename, 'w')
    for i in range(max_iter):
        log_file.write(f'{i} {process[i]}\n')
    for ii in range(len(t_para.variables_list)):
        log_file.write(
            f'{t_para.paramters_dir[ii]}={t_para.variables_list[ii]}\n')
    log_file.close()
    return t_para

def procedural_modelling():

    leaf_tex_filename = f'{data_path}/leaf.png'
    bark_tex_filename = f'{data_path}/bark.jpg'

    gpc_filename=f'{data_path}/data.txt'
    leaf_tex = cv2.imread(leaf_tex_filename, -1)
    bark_tex=cv2.imread(bark_tex_filename,-1)

    gpc_array, center, h = load_gpc(gpc_filename,Ground_h)

    para_mean, para_std=Control_Para_Aanlysis(gpc_array)
    optimal_para=Metropolis_Hastings_optimization(gpc_array,para_mean,para_std)
    _, dis, ts = from_para_to_tree_mesh(global_pool, gpc_array,
                                              para=optimal_para,
                                              this_sample_num=3)
    print(f'modelling error:{dis}')

    mesh = lpy.tree_mesh()
    mesh.leaf_ratio = leaf_tex.shape[0] / leaf_tex.shape[1]
    mesh.leaf_mean_size = 0.75
    mesh.leaf_drop_possibility = 0.15
    mesh.leaf_life_time = 7
    mesh.leaf_level = 1
    mesh.billboard_angles = 2
    mesh.LOD_MAX_LAYER = 3
    mesh.leaf_num = 1
    mesh.leaf_life_time = 10
    mesh.leaf_num_internode =2
    mesh.from_lpy(ts,leaf_tex, bark_tex, tex_width_real_length=0.5, tex_height_real_length=1.0,
                      compress_flag=True)
    h2 = mesh.get_max_bound()[2]
    mesh.scale(h / h2, np.array([0, 0, 0]))
    mesh.translate(np.array([center[0], center[1], 0]))
    best_mesh = mesh
    obj_filename = out_path + '/' + 'tree_model.obj'
    print('save tree model...')
    mesh.save_obj(obj_filename, leaf_tex_filename, bark_tex_filename)
    print('create LOD...')
    mesh.save_lod(out_path + '/lod_model', leaf_tex_filename, bark_tex_filename, -1)
def read_config_and_grow(path_filename):
    file = open(path_filename, 'r')
    lines = file.readlines()
    file.close()
    iters=0
    para = config.parameter()
    for i in range(len(para.variables_list)):
        name = lines[iters + i].split('=')[0]
        val = float(lines[iters + i].split(('='))[1])
        para.set(name, val)
        print(name, val)
    leaf_tex_filename = f'{data_path}/leaf.png'
    bark_tex_filename = f'{data_path}/bark.jpg'

    leaf_tex = cv2.imread(leaf_tex_filename, -1)
    bark_tex = cv2.imread(bark_tex_filename, -1)

    ts=lpy.derive(global_pool,para)


    mesh = lpy.tree_mesh()
    mesh.leaf_ratio = leaf_tex.shape[0] / leaf_tex.shape[1]
    mesh.leaf_mean_size = 0.75
    mesh.leaf_drop_possibility = 0.15
    mesh.leaf_life_time = 7
    mesh.leaf_level = 1
    mesh.billboard_angles = 2
    mesh.LOD_MAX_LAYER = 3
    mesh.leaf_num = 1
    mesh.leaf_life_time = 10
    mesh.leaf_num_internode = 2
    mesh.from_lpy(ts, leaf_tex, bark_tex, tex_width_real_length=0.5, tex_height_real_length=1.0,
                  compress_flag=True)
    obj_filename = out_path + '/' + 'tree_model.obj'
    print('save tree model...')
    mesh.save_obj(obj_filename, leaf_tex_filename, bark_tex_filename)
    print('create LOD...')
    mesh.save_lod(out_path + '/lod_model', leaf_tex_filename, bark_tex_filename, -1)
if __name__ == "__main__":
    #Ground_h=34.3
    if os.path.exists(f'{data_path}/{config_filename}'):
        read_config_and_grow(f'{data_path}/{config_filename}')
    else:
        procedural_modelling()


