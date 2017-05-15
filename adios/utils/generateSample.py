# -*- coding:utf-8 -*-

"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: generateSample.py
@time: 17/4/21 17:39
"""
import numpy as np


def sample(N):
    x_data, y_data = [], []
    w = np.random.normal(size=100)
    for i in range(N):
        y = np.zeros(4)
        x = np.random.random(100)
        if np.abs(np.dot(w, x)) <= 3.2:
            y[1:3] = 1
        else:
            y[0], y[-1] = 1, 1
        x = [float('%.3f' % xx) for xx in np.random.random(100)]
        x_data.append(x)
        y_data.append(y)

    return np.array(x_data), np.array(y_data)


if __name__ == "__main__":
    sample(N)
