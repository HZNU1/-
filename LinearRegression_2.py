import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_boston

class LinearRegression_2:
    def __init__(self):
        self._theta = None
    # 自变量x,因变量y,学习率learning_rate（默认0.0001），迭代次数（默认10000次）
    def fit_gd(self, x, y, learning_rate=0.0001, n_iters=1e4):
        #代价函数
        def J(theta, X, y):
            try:
                return np.sum((y - X.dot(theta)) ** 2) / (2*len(y))
            except:
                return float('inf')
        # 代价函数的偏导数
        def dJ(theta, X, y):
            return X.T.dot(X.dot(theta) - y) / len(y)
        #梯度下降
        def gradient_descent(X, y, initial_theta, learning_rate, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            #遍历
            while cur_iter < n_iters:
                gradient = dJ(theta, X, y)
                last_theta = theta
                theta = theta - learning_rate * gradient
                #收敛条件 本次迭代与上一次迭代之差小于epsilon=1e-8
                if (abs(J(theta, X, y) - J(last_theta, X, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        # 添加系数θ0，并且赋予它的参数为1
        X = np.hstack([np.ones((len(x), 1)), x])
        # 初始theta值为全为0的1×n数组
        initial_theta = np.zeros(X.shape[1])
        self._theta = gradient_descent(X, y, initial_theta, learning_rate, n_iters)
        return self

    def predict(self, X_predict):
        X = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X.dot(self._theta)

if __name__ == '__main__':
    # 从sklearn的数据集中导入特征data和房价target
    data, target = load_boston(return_X_y=True)
    data = preprocessing.scale(data)

    lr = LinearRegression_2()
    lr.fit_gd(data, target)
    # 提取一批样例观察一下拟合效果
    test = data[::50]
    test = np.mat(test)
    real = target[::50]  # 实际数值real
    estimate = np.array(lr.predict(test))  # 回归预测数值estimate
    # 打印结果
    for i in range(len(test)):
        print("实际值:", real[i], " 估计值:", estimate[0, i])