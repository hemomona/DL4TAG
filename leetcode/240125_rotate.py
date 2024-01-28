#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240125_rotate.py
# Time       ：2024/1/25 22:11
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：48. 旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
"""
from typing import List


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        我的思路：用
        [[ 5,  1,  9, 11],
         [ 2,  4,  8, 10],
         [13,  3,  6,  7],
         [15, 14, 12, 16]]
         作为例子手推即可
        """
        n = total_r = len(matrix)
        while n > 1:
            # 记录处理到第几层
            r = (total_r - n) // 2
            # 将第一行 从左到右 与最后一列 从上到下 换位
            for i in range(0, n):
                first_row = r
                last_col = total_r - 1 - r
                matrix[first_row][r + i], matrix[r + i][last_col] = matrix[r + i][last_col], matrix[first_row][r + i]
            # 将第一行 从左到右 与最后一行 从右到左 换位
            for i in range(0, n):
                first_row = r
                last_row = total_r - 1 - r
                matrix[first_row][r + i], matrix[last_row][total_r - 1 - i - r] = \
                    matrix[last_row][total_r - 1 - i - r], matrix[first_row][r + i]
            # 将第一列（不含最后一个） 从上到下 与第一行（不含第一个） 从右到左 换位
            for i in range(0, n - 1):
                first_col = r
                first_row = r
                matrix[r + i][first_col], matrix[first_row][total_r - 1 - i - r] = \
                    matrix[first_row][total_r - 1 - i - r], matrix[r + i][first_col]
            # 内一层
            n = n - 2


s = Solution()
m = [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]
s.rotate(m)


class Solution:
    """
    原地旋转：
    以位于矩阵四个角点的元素为例，设矩阵左上角元素 A 、右上角元素 B 、右下角元素 C 、左下角元素 D 。
    矩阵旋转 90º 后，相当于依次先后执行 D→A, C→D, B→C, A→B 修改元素，即「首尾相接」的元素旋转操作
    """
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n // 2):
            # 通过列数加一，确保中位数都旋转了
            for j in range((n + 1) // 2):
                tmp = matrix[i][j]
                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = tmp


class Solution:
    """
    两次翻转：
    先将其通过水平轴翻转，再通过 左上-右下 主对角线翻转，即等价于 顺时针旋转90°
    """
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        # 主对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
