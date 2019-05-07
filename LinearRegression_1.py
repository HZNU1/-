# coding=utf-8
import numpy as np
from sklearn.datasets import load_boston  # 导入sklearn包中波士顿房价数据集


class LinerRegression_1:

    def __init__(self,x,y):
        self.x = np.mat(x)
        self.y = np.mat(y)


    def regression(self):
        # 添加系数θ0，并且赋予它的参数为1
        theta0 = np.ones((len(data), 1))
        self.x = np.hstack((self.x, theta0))
        x_T = self.x.T  # 计算X矩阵的转置矩阵
        self.theta = (x_T * self.x).I * x_T * self.y.T  # 由最小二乘法计算得出的参数向量

    def predict(self, vec):
        vec = np.mat(vec)
        vec0 = np.ones((len(vec), 1))
        vec = np.hstack((vec, vec0))
        estimate = np.matmul(vec, self.theta) #点乘
        return estimate


if __name__ == '__main__':
    # 从sklearn的数据集中导入特征data和房价target
    data, target = load_boston(return_X_y=True)

    lr = LinerRegression_1(data, target)
    lr.regression()
    # 提取一批样例观察一下拟合效果
    test = data[::50]
    M_test = np.mat(test)
    real = target[::50]  # 实际数值real
    estimate = np.array(lr.predict(M_test))  # 回归预测数值estimate
    # 打印结果
    print(estimate)
    for i in range(len(test)):
        print("实际值:", real[i], " 估计值:", estimate[i, 0])