#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240121_findKthLargest.py
# Time       ：2024/1/21 23:47
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：215. 数组中的第K个最大元素
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
"""
import random


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quickSelect(arr, left, kth, right):
            """
            快速排序：从子数组a[l...r]中任意选择一个元素x，调整子数组使得左边元素大于它，右边元素小于它，递归调用。
            每次划分我们一定可以确定选择的x的最终位置，那么只要它的位置q为第k个，我们返回a[q]即可，如果q小于目标位置则递归右子区间，如果q大于目标位置则递归左子区间。
            快排的性能与划分的子数组的长度相关。如果每次都划分为1和n-1，那么时间代价为O(n^2)，如果随机化则为O(n)。
            :param arr:
            :param left:
            :param kth:
            :param right:
            :return:
            """
            curr = partition(arr, left, right)
            if curr == kth:
                return arr[curr]
            elif curr < kth:
                return quickSelect(nums, curr + 1, kth, right)
            else:
                return quickSelect(nums, left, kth, curr - 1)

        def partition(arr, left, right):
            # 我们不关心这次划分左右是否有序，只关心arr[pivot]最终定位到哪个位置
            pivot = random.randint(left, right)
            swap(arr, pivot, right)
            '''
            index记录大于等于arr[right]的所有数字的最大索引
            当遇到小于arr[right]的数字arr[j]时，index不变，仍等于j-1，j-1及之前的数字均大于等于arr[right]
            再遇到下一个大于等于arr[right]的数字arr[x]时，index=j，将arr[j]与arr[x]交换位置
            直到遍历完毕，将arr[right]与均大于等于arr[right]的数字的下一位交换位置，即为pivot在最终数组的位置
            '''
            index = left - 1
            for j in range(left, right):
                if arr[j] >= arr[right]:
                    index = index + 1
                    swap(arr, index, j)
            index = index + 1
            swap(arr, index, right)
            return index

        def swap(arr, i, j):
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp

        return quickSelect(nums, 0, k - 1, len(nums) - 1)


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """heapsort堆排
        建堆：将待排序的数组初始化为大根堆。此时，根节点即为整个数组中的最大值。
        交换和调整：将堆顶元素与末尾元素进行交换，此时末尾即为最大值。除去末尾元素后，将其他n-1个元素重新构造成一个大根堆，如此便可得到原数组n个元素中的次大值。
        重复步骤二，直至堆中仅剩一个元素，如此便可得到一个有序序列了。
        将一个完全二叉树构造成一个大顶堆的一个实现方式是从最后一个「非叶子节点」为根节点的子树出发，从右往左、从下往上进行调整操作。
        若完全二叉树从 0 开始进行编号，则第一个非叶子节点为n/2−1；若完全二叉树从 1 开始进行编号，则第一个非叶子节点为n/2。
        对于以某个非叶子节点的子树而言，其基本的调整操作包括:
            如果该节点大于等于其左右两个子节点，则无需再对该节点进行调整，因为它的子树已经是堆有序的；
            如果该节点小于其左右两个子节点中的较大者，则将该节点与子节点中的较大者进行交换，并从刚刚较大者的位置出发继续进行调整操作，直至堆有序。
        """

        def buildMaxHeap(arr):
            n = len(arr)
            for root in range(n // 2 - 1, -1, -1):
                adjustChild(arr, root, n - 1)

        def adjustChild(arr, root, lastLeaf):
            if root > lastLeaf:
                return
            t = arr[root]
            child = 2 * root + 1
            while child <= lastLeaf:
                if child + 1 <= lastLeaf and arr[child] < arr[child + 1]:
                    child = child + 1
                # 下面这个判断与前一判断并列，t如果大于等于左右子节点，将t赋给该节点（root或上轮的child）即可
                if t >= arr[child]:
                    break
                arr[root] = arr[child]
                root = child
                child = 2 * root + 1
            arr[root] = t

        n = len(nums)
        buildMaxHeap(nums)
        for i in range(k - 1):
            # python称为序列解包，可将=右边视为构建了个新的列表，将元素依次赋给左边
            # 即将下标i大的数放在下标n-1-i的节点
            nums[0], nums[n - 1 - i] = nums[n - 1 - i], nums[0]
            # 再调整0到n-1-i-1这颗树为大根堆
            adjustChild(nums, 0, n - 1 - i - 1)
        return nums[0]


