#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240202_sortArray.py
# Time       ：2024/2/2 11:00
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：912. 排序数组
给你一个整数数组 nums，请你将该数组升序排列。
三路快排：https://leetcode.cn/problems/sort-an-array/solution/by-cao-chi-yhxq/
"""
import random
from typing import List


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) == 0:
            return nums
        return self.quickSort_3ways(nums, 0, len(nums) - 1)

    def quickSort(self, arr: List[int], left: int, right: int) -> List[int]:
        if left < right:
            pivot = self.partition(arr, left, right)
            self.quickSort(arr, left, pivot - 1)
            self.quickSort(arr, pivot + 1, right)
        return arr

    def quickSort_3ways(self, arr: List[int], left: int, right: int) -> List[int]:
        if left < right:
            left_index, right_index = self.partition_3ways(arr, left, right)
            self.quickSort(arr, left, left_index)
            self.quickSort(arr, right_index, right)
        return arr

    def partition(self, arr: List[int], left: int, right: int) -> int:
        # randint = [a, b]
        pivot = random.randint(left, right)
        arr[pivot], arr[right] = arr[right], arr[pivot]
        index = left - 1
        for i in range(left, right):
            if arr[i] < arr[right]:
                index += 1
                arr[i], arr[index] = arr[index], arr[i]
        index += 1
        arr[right], arr[index] = arr[index], arr[right]
        return index

    def partition_3ways(self, arr: List[int], left: int, right: int) -> (int, int):
        """
        三路快排，解决快速排序对于重复元素过多的数组处理很慢，left_index记录小于基元的最大下标，right_index记录大于基元的最小下标
        """
        pivot = random.randint(left, right)
        arr[pivot], arr[right] = arr[right], arr[pivot]
        index, left_index, right_index = left + 1, left, right - 1
        # <= 非常关键
        while index <= right_index:
            if arr[index] < arr[right]:
                arr[left_index], arr[index] = arr[index], arr[left_index]
                left_index += 1
                index += 1
            elif arr[index] > arr[right]:
                arr[right_index], arr[index] = arr[index], arr[right_index]
                right_index -= 1
            else:
                index += 1
        arr[right], arr[right_index] = arr[right_index], arr[right]
        return left_index-1, right_index+1


nums = [-4,0,7,4,9,-5,-1,0,-7,-1]
s = Solution()
a = s.sortArray(nums)
print(a)
