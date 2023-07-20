#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : multinomial_sample.py
# Time       ：2023/7/20 10:42
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
         https://blog.csdn.net/liangzuojiayi/article/details/78183783
         https://zhuanlan.zhihu.com/p/75045335
"""
# 在使用jupyter notebook 或者 jupyter qtconsole的时候，才会经常用到%matplotlib
# %matplotlib inline
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)  # 相对频率作为估计值

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# counts = Tensor.size(500,6)
# torch.cumsum(input, dim=?) or input.cumsum(dim=?)
# dim = 0按行累加
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()
