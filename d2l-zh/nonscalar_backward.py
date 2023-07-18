#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : nonscalar_backward.py
# Time       ：2023/7/18 22:37
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
            https://blog.csdn.net/qq_43655233/article/details/112184267
"""

# 非标量反向传播
import torch
# 1、定义叶子节点及计算节点
# 定义叶子节点张量x
x = torch.tensor([2, 3], dtype=torch.float, requires_grad=True)
# 初始化雅克比矩阵
J = torch.zeros(2, 2)
# 初始化目标张量，形状为1×2
y = torch.zeros(1, 2)
# 定义y与x之间的映射关系
y[0, 0] = x[0] ** 2 + 3 * x[1]
y[0, 1] = x[1] ** 2 + 2 * x[0]
# 2、调用backward来获取y对x的梯度
# 注释下一行以运行第3步的backward
# y.backward(torch.Tensor([[1, 1]]))
print(x.grad)
# 结果显然是错误的
# tensor([6., 9.])

# 3、正确计算张量对张量求导
# 生成y1对x的梯度
# 注释该段以运行第4步的backward
y.backward(torch.Tensor([[1, 0]]), retain_graph=True)
J[0] = x.grad
print(x.grad)
x.grad = torch.zeros_like(x.grad)
# 生成y2对x的梯度
y.backward(torch.Tensor([[0, 1]]))
J[1] = x.grad
print(x.grad)
# 显示雅克比矩阵的值
print(J)
# tensor([4., 3.])
# tensor([2., 6.])
# tensor([[4., 3.],
#         [2., 6.]])

# 4、d2l autograd章节的非标量变量的反向传播
# y.sum().backward()
# print(x.grad)
# tensor([6., 9.])
# 说明 如果不指定gradient，默认就会把y加起来反向传播

