# encoding: utf-8
import numpy as np
import random
from scipy.spatial import KDTree
import time
import sklearn.cluster  # import DBSCAN


class visitlist:
    def __init__(self, count=0):
        self.unvisitedlist = [i for i in range(count)]
        self.visitedlist = list()
        self.unvisitednum = count

    def visit(self, pointId):
        self.visitedlist.append(pointId)#已经访问的点 增加
        self.unvisitedlist.remove(pointId)#未访问的点 减少
        self.unvisitednum -= 1 #未访问的点的数量


class cluster:
    def __init__(self, ctype):
        self.ctypr = ctype
        self.points = list()


def DBSCAN(X: np.ndarray, r: float, minPts: int):
    pointnum = X.shape[0]
    v = visitlist(pointnum)
    clustersSet = list()
    noise = cluster(-1)
    tree = KDTree(X)
    k = 0

    while v.unvisitednum > 0:
        randid = random.choice(v.unvisitedlist)
        v.visit(randid)
        N = tree.query_ball_point(X[randid], r)
        if len(N) < minPts:
            noise.points.append(randid)
        else:
            clus = cluster(k)
            clus.points.append(randid)
            N.remove(randid)
            while len(N) > 0:
                p = N.pop()
                if p in v.unvisitedlist:
                    v.visit(p)
                    clus.points.append(p)
                    pN = tree.query_ball_point(X[p], r)
                    if len(pN) >= minPts:
                        pN.remove(p)
                        N = N + pN
            clustersSet.append(clus)

    clustersSet.append(noise)
    return clustersSet

if __name__ == "__main__":
    import time
    from sklearn import datasets
    import matplotlib.pyplot as plt

    centers = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
    pointlala, labelsTrue = datasets.make_blobs(n_samples=600, centers=centers, cluster_std=0.4, random_state=0)
    C = DBSCAN(pointlala, 0.3, 10)

    plt.scatter(pointlala[:, 0], pointlala[:, 1])
    trys1 = pointlala[C[0].points]
    plt.scatter(trys1[:, 0], trys1[:, 1])
    trys2 = pointlala[C[1].points]
    plt.scatter(trys2[:, 0], trys2[:, 1])
    trys3 = pointlala[C[2].points]
    plt.scatter(trys3[:, 0], trys3[:, 1])
    trys4 = pointlala[C[3].points]
    plt.scatter(trys4[:, 0], trys4[:, 1])

    plt.show()