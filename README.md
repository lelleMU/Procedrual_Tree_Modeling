# Procedrual_Tree_Modeling
An implementation of [**Oblique Photogrammetry Supporting Procedural Tree Modeling in Urban Areas**](https://www.sciencedirect.com/science/article/pii/S0924271623001259) using python3.9.
# Introduction:
Trees are common natural objects widely distributed in urban areas. The modeling quality of trees has significant visual effects for urban modeling. Tree models directly from the oblique photogrammetry pipeline are erratic because tree textures are repeating and confusing. Existing tree modeling approaches either cannot deal with the input data of poor quality or are time-consuming. This paper presents an oblique photogrammetry-supporting procedural tree modeling approach to solve this problem. Specifically, we propose a point cloud data-supported hybrid parametric model that models trees by simulating the growth of natural trees. To solve the challenging optimization problem in high dimensional space defined by the parametric model, a Control Parameter Analysis (CPA)-based optimization method is proposed to find the approximate solution with the highest resemblance to the input data. Finally, an automatic level of detail (LOD) control method of tree models is proposed to form a complete workflow for rendering the urban scenes. Experimental results indicate that the proposed method can generate procedural tree models in urban areas with satisfying accuracy and efficiency. The mean normalized distance between the generated tree models and input data in the test regions is 0.022, and it takes about ten minutes to generate each tree model.

# Dependency:
    Numpy
    open3d
    numba
    opencv

# DATA PREPARATION:
create a data path containing:
* Your tree points data:
    * Data name:data.txt 
    * Data format:[x,y,z]
* Leaf texture image:
    * Image name: leaf.png
* Bark texture image:
    * Image name: bark.jpg

# Perform modelling:
    python modelling.py [-h] [--DATA_PATH DATA_PATH] [--OUT_PATH OUT_PATH] [--GROUND_H GROUND_H]
                    [--MAX_ITERATION MAX_ITERATION]
Remember setting the `GROUND_H`. By default, it is the lowest point height of the input point cloud minus 3.
`MAX_ITERATION` is the maximum number of iterations. It is 100 by default.

# Output:
    tree_model.obj: Tree model
    log: A path containing a LOG.txt, which contains the optimization process and the optimal parameters
    lod_model: A path containing models with different levels of detail
We recommend using [**Blender**](https://www.blender.org/) for model visualization.

# Modelling using predefined parameters:
It is allowed to conduct procedural modelling without reference points, but only with predefined parameters.

   `python modelling.py [-h] [--DATA_PATH DATA_PATH] [--OUT_PATH OUT_PATH] [--CONFIG_FILENAME CONFIG_FILENAME]`
   
Make sure that you have a configure file in the data path. An example of the configure file is as follows:

```
NLB=1.0
BAM=18.452812325156984
RAM=150.0
AB_D=0.26611275440942744
AD_BF=0.2
AD_LF=0.5
AC=0.40594897011136666
AC_AF=1.01103261826377
AC_LF=1.08042313872564
LB_PF=0.1942775166584318
BDM=40.0
t=17.948906184831458
LRR=2.5
shedding_factor=0.01
v_tropism_w=0.05
GBF=10.0
GBA=0.1
IL_LF=0.9751801948267683
FP_Y=0.30106929015307793
FP_AF=1.0
FP_A=15.0
MAX_L=4.520881602774618
IL_AF=0.94
IBL=0.5
LRR_AF=0.96
light_sensitiveness=0.1
```
Make sure you get all the names right, and donot forget the texture images!

# Citation:
```
@article{WANG2023120,
title = {Oblique photogrammetry supporting procedural tree modeling in urban areas},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {200},
pages = {120-137},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2023.05.008},
url = {https://www.sciencedirect.com/science/article/pii/S0924271623001259},
author = {Xuan Wang and Hanyu Xiang and Wenyuan Niu and Zhu Mao and Xianfeng Huang and Fan Zhang},
keywords = {Oblique photogrammetry, Procedural modeling, Inverse procedural modeling, L-system, Parametric model, Tree modeling, Metropolis-Hastings},
abstract = {Trees are common natural objects widely distributed in urban areas. The modeling quality of trees has significant visual effects for urban modeling. Tree models directly from the oblique photogrammetry pipeline are erratic because tree textures are repeating and confusing. Existing tree modeling approaches either cannot deal with the input data of poor quality or are time-consuming. This paper presents an oblique photogrammetry-supporting procedural tree modeling approach to solve this problem. Specifically, we propose a point cloud data-supported hybrid parametric model that models trees by simulating the growth of natural trees. To solve the challenging optimization problem in high dimensional space defined by the parametric model, a Control Parameter Analysis (CPA)-based optimization method is proposed to find the approximate solution with the highest resemblance to the input data. Finally, an automatic level of detail (LOD) control method of tree models is proposed to form a complete workflow for rendering the urban scenes. Experimental results indicate that the proposed method can generate procedural tree models in urban areas with satisfying accuracy and efficiency. The mean normalized distance between the generated tree models and input data in the test regions is 0.022, and it takes about ten minutes to generate each tree model. Our code is available at https://github.com/lelleMU/Procedrual_Tree_Modeling.}
}
```
We hope you find our work helpful or interesting! :)



    

