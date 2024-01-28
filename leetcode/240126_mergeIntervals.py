#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240126_mergeIntervals.py
# Time       ：2024/1/26 20:45
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：56. 合并区间
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
排序
"""
from typing import List


class Solution:
    """
    屎代码，难以修改
    """

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        new_intervals = []
        has_merged = [0] * len(intervals)
        for i in range(len(intervals)):
            merged_start = [intervals[i][0]]
            merged_end = [intervals[i][1]]
            modify_new = False

            for j in range(i + 1, len(intervals)):
                if intervals[j][0] > max(merged_end) or intervals[j][1] < min(merged_start):
                    continue

                if has_merged[j]:
                    modify_new = True
                    break

                has_merged[j] = len(new_intervals) + 1
                merged_start.append(intervals[j][0])
                merged_end.append(intervals[j][1])
            if modify_new:
                index = has_merged[j] - 1
                has_merged[i] = index + 1
                new_intervals[index] = [min(intervals[i][0], new_intervals[index][0]),
                                        max(intervals[i][1], new_intervals[index][1])]
            elif not has_merged[i]:
                new_intervals.append([min(merged_start), max(merged_end)])
            else:
                pass
        return new_intervals


s = Solution()
print(s.merge([[2, 3], [4, 6], [5, 7], [3, 4]]))


class Solution:
    """
    如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。
    如果当前区间的左端点在数组 merged 中最后一个区间的右端点之后，那么它们不会重合，我们可以直接将这个区间加入数组 merged 的末尾；
    否则，它们重合，我们需要用当前区间的右端点更新数组 merged 中最后一个区间的右端点，将其置为二者的较大值
    """
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加；否则与上一区间合并
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
