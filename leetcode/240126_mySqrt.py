#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240126_mySqrt.py
# Time       ：2024/1/26 10:54
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：69. x 的平方根
给你一个非负整数 x ，计算并返回 x 的 算术平方根 。
由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。
注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。
"""


class Solution:
    """
    基本相当于穷举
    """
    def mySqrt(self, x: int) -> int:
        if x == 1:
            return 1
        i = 0
        for i in range(x//2+1):
            if i*i == x:
                return i
            if i*i > x:
                return i-1
        return i


class Solution:
    """
    我的思路：利用2的指数找到x的上下界，然后指数整除以2，从上界找到下界
    """
    def mySqrt(self, x: int) -> int:
        if x == 0 or x == 1:
            return x

        # 先定位到最小使得 2^i >= x 的指数i
        power = 1
        exponent = 0
        for i in range(x):
            if power >= x:
                exponent = i
                break
            else:
                power *= 2

        # exp为11，则power = 2^6; exp为10，则power = 2^5
        for i in range(exponent // 2):
            # 须是// 否则power为float
            power //= 2

        # end须减一，避免n一直大于x
        for i in range(power, power // 2 - 1, -1):
            n = i * i
            if n <= x:
                return i


s = Solution()
print(s.mySqrt(8))


class Solution:
    """
    x^0.5 = (e^(lnx))^0.5 = e^(0.5lnx)
    因为指数函数和对数函数返回浮点数，因此会有误差
    """
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        ans = int(math.exp(0.5 * math.log(x)))
        return ans+1 if (ans+1)**2 <= x else ans


class Solution:
    """
    二分查找
    """
    def mySqrt(self, x: int) -> int:
        l, r, ans = 0, x, -1
        while l <= r:
            mid = (l + r) // 2
            if mid * mid <= x:
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans


class Solution:
    """
    牛顿迭代法：
    C的平方根为函数 f(x) = x^2 - C的零点
    取x0作为初始值，过(x0, x0^2-C)作斜率为2x0的导线，则该导线的零点相较于x0，会更接近于f(x)的零点
    该导线函数为 f(x) - (x0^2-C) = 2x0(x-x0)，解得零点为0.5*(x0+C/x0)
    选择x0 = C作为初始值，因为我们要求正平方根
    当相邻两次结果小于一个极小值1e-7，我们就可以认为找到零点了
    """
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0

        C, x0 = float(x), float(x)
        while True:
            xi = 0.5 * (x0 + C / x0)
            if abs(x0 - xi) < 1e-7:
                break
            x0 = xi
        return int(x0)
