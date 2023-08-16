'''
A script to create a leaf texture image using user-defined parameters
One can also create a leaf texture image using an existing leaf image
'''

import numpy as np
import cv2
import Utility
from math import *
class leaf_parameter:
    leaf_size=50
    stem_size=20
    ba_ratio=0.5
    leaf_color=np.array([44,111,78,255])
    stem_color= (25, 81, 140, 255)
    N = 0
    N1 = 0
    angle = 60
    angle2=30
    level_ratio = sqrt(2)
    level_reduce=False
    inter_l=2*leaf_size
    def __init__(self):
        return

def get_oval_set(f1,f2,a,pf=0.0):
    min_x=int(min(f1[0],f2[0]))
    max_x=int(max(f1[0],f2[0]))
    min_y = int(min(f1[1], f2[1]))
    max_y = int(max(f1[1], f2[1]))
    set=[]
    for x in range(min_x-a,max_x+a):
        for y in range(min_y-a,max_y+a):
            dis1=Utility.distance(f1,np.array([x,y]))
            dis2=Utility.distance(f2,np.array([x,y]))
            if (dis1+dis2)<=2*a:
                set.append((x,y))
    return np.array(set)
def grow_leaf(image,p,dir,leaf_stem_l,leaf_stem_width,a,c,leaf_color,pointy_factor=0.0):
    this_pt1 = p
    this_pt2 = (this_pt1 + dir * leaf_stem_l).astype(int)
    cv2.line(image, this_pt1, this_pt2, (25, 81, 140, 255), leaf_stem_width)
    oval_f1 = this_pt2 + dir * (a - c)
    oval_f2 = this_pt2 + dir * (a + c)
    this_oval = get_oval_set(oval_f1, oval_f2, a)
    if len(this_oval)>0:
        image[this_oval[:, 1], this_oval[:, 0]] = leaf_color

def get_R(angle):
    R = np.zeros([2, 2])
    R[0, 0] = cos(angle * pi / 180)
    R[0, 1] = sin(angle * pi / 180)
    R[1, 0] = -sin(angle * pi / 180)
    R[1, 1] = cos(angle * pi / 180)
    return R
def creat_leaf(para:leaf_parameter):#a 长轴 b 短轴 N 叶子层数
    a=para.leaf_size
    level_ratio=para.level_ratio
    ba_ratio=para.ba_ratio
    angle=para.angle
    angle2=para.angle2
    N=para.N
    N1=para.N1
    leaf_color=para.leaf_color

    stem_color=para.stem_color
    b=round(ba_ratio*a)
    stem_width=round(b/16.)
    c=sqrt(a*a-b*b)

    a_n = round(a / level_ratio)
    c_n = round(c / level_ratio)

    a_nn= round(a_n / level_ratio)
    c_nn = round(c_n / level_ratio)
    R=get_R(angle)
    R2=get_R(-angle)

    R_n=get_R(angle2)
    R2_n=get_R(-angle2)

    #leaf_stem_l=b
    leaf_stem_l = 0
    leaf_stem_width=max(round(stem_width/2.),1)
    inter_l=para.inter_l
    stem_l = para.stem_size
    H=2000
    W=2000
    half_w=round(W/2)
    image=np.zeros([H,W,4])
    bottom_pt1=np.array([half_w,0]).astype(int)
    top_leaf_pt2=np.array([half_w,stem_l+N*inter_l]).astype(int)
    cv2.line(image,bottom_pt1,top_leaf_pt2,stem_color,1)
    for i in range(N):

        this_N1 = max(N1 - i, 0) if para.level_reduce else N1
        h = stem_l + i * inter_l
        dir =R[:,1]
        this_level_l = stem_l /level_ratio + inter_l / level_ratio * this_N1
        pt1 = np.array([half_w, h]).astype(int)
        pt2 = (pt1 + dir * this_level_l).astype(int)
        cv2.line(image, pt1, pt2,stem_color, 1)

        for j in range(this_N1):
            hj=stem_l /level_ratio+inter_l/level_ratio*j
            this_R=R.dot(R_n)
            this_dir=this_R[:,1]
            this_pt1=(pt1+dir*hj).astype(int)
            grow_leaf(image,this_pt1,this_dir,leaf_stem_l/level_ratio,leaf_stem_width,a_nn,c_nn,leaf_color)

            this_R = R.dot(R2_n)
            this_dir = this_R[:, 1]
            grow_leaf(image, this_pt1, this_dir, leaf_stem_l / level_ratio, leaf_stem_width, a_nn,
                      c_nn, leaf_color)
        grow_leaf(image,pt2,dir,0,leaf_stem_width,a_n,c_n,leaf_color)

        dir = R2[:, 1]
        this_level_l = stem_l / level_ratio + inter_l / level_ratio * this_N1
        pt1 = np.array([half_w, h])
        pt2 = (pt1 + dir * this_level_l).astype(int)
        cv2.line(image, pt1, pt2,stem_color, 1)
        for j in range(this_N1):
            hj = stem_l / level_ratio + inter_l / level_ratio * j
            this_R = R2.dot(R_n)
            this_dir = this_R[:, 1]
            this_pt1 = (pt1 + dir * hj).astype(int)
            grow_leaf(image, this_pt1, this_dir, leaf_stem_l / level_ratio, leaf_stem_width, a_nn,
                     c_nn, leaf_color)

            this_R = R2.dot(R2_n)
            this_dir = this_R[:, 1]
            grow_leaf(image, this_pt1, this_dir, leaf_stem_l / level_ratio, leaf_stem_width, a_nn,
                      c_nn, leaf_color)
        grow_leaf(image, pt2, dir, 0, leaf_stem_width, a_n, c_n, leaf_color)
    grow_leaf(image,top_leaf_pt2,np.array([0,1]),0,leaf_stem_width,a,c,leaf_color)
    img=np.flip(image,axis=0)
    pixels_x,pixels_y=np.where(img[:,:,3]==255)
    pixels_x_min=np.min(pixels_x)
    pixels_x_max=np.max(pixels_x)
    pixels_y_min = np.min(pixels_y)
    pixels_y_max = np.max(pixels_y)
    return img[pixels_x_min:pixels_x_max+1,pixels_y_min:pixels_y_max+1]

def create_from_lib(filename,color):
    leaf_tex=cv2.imread(filename,-1)
    leaf_tex[np.where(leaf_tex[:,:,3]>0)]=color
    return leaf_tex


