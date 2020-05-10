# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from RANSAC import *
import open3d as o3d
import open3d
from pyntcloud import PyntCloud
from pandas import DataFrame
import sklearn.cluster
from DBSCAN import *
from mpl_toolkits.mplot3d import Axes3D

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    #调用实现的RANSAC方法，详见另一个RANSAC.py
    planeids = PlaneRANSAC(data,0.35)
    segmengted_cloud = data[planeids]#由上一步得到的地面点的索引进行取值

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud,planeids

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    #使用sklearn中的聚类
    Css = sklearn.cluster.DBSCAN(eps=0.50, min_samples=4).fit(data)
    clusters_index = np.array(Css.labels_)

    #使用自己实现的聚类
    # clusters_index = DBSCAN(data,0.5,100)
    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = './data/' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    # iteration_num = len(cat)
    iteration_num = 1
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points,planeids = ground_segmentation(data=origin_points)
        planepcd = o3d.geometry.PointCloud()
        planepcd.points = o3d.utility.Vector3dVector(segmented_points)

        c = [0, 0, 255]
        cs = np.tile(c, (segmented_points.shape[0], 1))
        planepcd.colors = o3d.utility.Vector3dVector(cs)

        othersids = []
        for i in range(origin_points.shape[0]):
            if i not in planeids:
                othersids.append(i)
        otherdata = origin_points[othersids]
        otherpcd = o3d.geometry.PointCloud()
        otherpcd.points = o3d.utility.Vector3dVector(otherdata)
        c = [255, 0, 0]
        cs = np.tile(c, (otherdata.shape[0], 1))
        otherpcd.colors = o3d.utility.Vector3dVector(cs)
        o3d.visualization.draw_geometries([planepcd, otherpcd])

        cluster_index = clustering(otherdata)#对于非地面的点云进行聚类
        colorset = [[128, 128, 0], [0, 128, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [128, 0, 0]]
        print(len(cluster_index))
        point_draw = []
        for index in set(cluster_index):

            point_index = np.where(cluster_index == index)
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(otherdata[point_index])

            c = colorset[index % 6]
            if index == -1:
                c = [0, 0, 0]

            cs = np.tile(c, (otherdata[point_index].shape[0], 1))
            cloud.colors = open3d.utility.Vector3dVector(cs)
            point_draw.append(cloud)

        point_draw.append(planepcd)
        open3d.visualization.draw_geometries(point_draw)
        # point_draw = []
        # for cluser in cluster_index:
        #     ppk = open3d.geometry.PointCloud()
        #     ppk.points = open3d.utility.Vector3dVector(otherdata[kaka])
        # plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
