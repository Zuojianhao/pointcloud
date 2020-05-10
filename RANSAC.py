# encoding: utf-8

import numpy as np
import random
import math
import sys

import numpy as np
import random


def PlaneLeastSquare(X: np.ndarray):
    # z=ax+by+c,return a,b,c
    A = X.copy()
    b = np.expand_dims(X[:, 2], axis=1)
    A[:, 2] = 1

    # 通过X=(AT*A)-1*AT*b直接求解
    #ax + by + c = z
    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    x = np.dot(A3, b)
    return x

def PlaneRANSAC(X: np.ndarray, tao: float, e=0.4, N_regular=100):
    # return plane ids
    s = X.shape[0]

    count = 0
    p = 0.99
    dic = {}

    # 确定迭代次数
    if math.log(1 - (1 - e) ** s) < sys.float_info.min:
        N = N_regular
    else:
        N = math.log(1 - p) / math.log(1 - (1 - e) ** s)

    # 开始迭代
    while count < N:

        ids = random.sample(range(0, s), 3)
        p1, p2, p3 = X[ids]
        # 判断是否共线
        L = p1 - p2
        R = p2 - p3
        if 0 in L or 0 in R:
            continue
        else:
            if L[0] / R[0] == L[1] / R[1] == L[2] / R[2]:
                continue

        # 计算平面参数
        a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
        b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
        c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
        d = 0 - (a * p1[0] + b * p1[1] + c * p1[2]);

        dis = abs(a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d) / (a ** 2 + b ** 2 + c ** 2) ** 0.5

        # 计算inline的点
        idset = []
        for i, d in enumerate(dis):
            if d < tao:
                idset.append(i)

        # 再使用最小二乘法
        p = PlaneLeastSquare(X[idset])
        a, b, c, d = p[0], p[1], -1, p[2]

        dic[len(idset)] = [a, b, c, d]

        if len(idset) > s * (1 - e):#如果inline的点超过预想的总点数的比例的数量，则停止循环
            break

        count += 1

    parm = dic[max(dic.keys())]
    a, b, c, d = parm
    dis = abs(a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d) / (a ** 2 + b ** 2 + c ** 2) ** 0.5

    idset = []
    for i, d in enumerate(dis):#选出地面上的点
        if d < tao:
            idset.append(i)
    return np.array(idset)