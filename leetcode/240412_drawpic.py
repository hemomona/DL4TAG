#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240412_drawpic.py
# Time       ：2024/4/12 12:10
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt


def f(x, pd, t, v0):
    # numpy提供了对数组进行元素级运算的函数版本。
    return (v0 - (1 + t * x) * v0 * np.exp(-t * x)) / (pd - t * v0 * np.exp(-t * x))


def f2(x, pd, t, v0):
    # numpy提供了对数组进行元素级运算的函数版本。
    return pd * x + v0 * np.exp(-t * x) - v0


# 定义x的值：从-10到400，总共2000个点
x = np.linspace(-10, 400, 2000)
# 计算y值
y = f2(x, 5840, 60, 0.53)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制函数
plt.plot(x, y, label='5840 60 0.53')

# 添加标题和标签
plt.title('Function Plot')
plt.xlabel('x')
plt.ylabel('f(x)')

# 添加图例
plt.legend()

# 显示网格（可选）
plt.grid(True)

# 显示图形
plt.show()



