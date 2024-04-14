#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240411_zxy_Kobs.py
# Time       ：2024/4/11 19:55
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
"""
import math
import numpy as np
import pandas as pd

"""
牛顿迭代法：
Kobs为函数 f(x) = (P-D)x + v0 * exp(-t * x) - v0的零点
取x0作为初始值，过(x0, f(x0))作斜率为 P-D - t * v0 * exp(-t * x0)的导线，则该导线的零点相较于x0，会更接近于f(x)的零点
该导线函数为 y - f(x0) = k(x-x0)，解得零点
选择x0 = P-D作为初始值，因为我们要求正平方根
当相邻两次结果小于一个极小值1e-7，我们就可以认为找到零点了
"""


def myKobs(f: int, t: int, v0: float) -> float:
    if f == 0:
        return 0
    m = 1e-7
    x0 = 1
    try:
        while True:
            # numpy 的 exp 函数比 math.exp 更健壮，能够更好地处理大范围的输入
            xi = (v0 - (1 + t * x0) * v0 * np.exp(-t * x0)) / (f - t * v0 * np.exp(-t * x0))
            if abs(x0 - xi) < m:
                break
            x0 = xi
        return x0
    except OverflowError:
        print(t, "时间计算溢出")
        return None


def process_excel(input_file, output_file, v0):
    # 读取无表头xlsx文件
    df = pd.read_excel(input_file, header=None)
    if df.shape[1] < 3:
        raise ValueError("输入的Excel文件中至少需要3列数据，第一列为t，第三列为P-D")

    print('正在计算', input_file)
    # 使用apply函数来应用自定义函数myKobs到每一行
    df['kobs'] = df.apply(lambda row: myKobs(row.iloc[2], row.iloc[0], v0), axis=1)
    df.to_excel(output_file, index=False)
    print('已保存到', output_file)


def compute_mean(input_file):
    df = pd.read_excel(input_file)
    # IQR
    # Calculate the upper and lower limits
    Q1 = df['kobs'].quantile(0.25)
    Q3 = df['kobs'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(df['kobs'] >= upper)[0]
    lower_array = np.where(df['kobs'] <= lower)[0]

    # Removing the outliers
    df.drop(index=upper_array, inplace=True)
    df.drop(index=lower_array, inplace=True)
    print('IQR mean: ', np.mean(df['kobs']))


if __name__ == '__main__':
    v0 = 161
    input_file = './zxy.xlsx'
    output_file = './test_e-7.xlsx'
    # process_excel(input_file, output_file, v0)
    compute_mean(output_file)
