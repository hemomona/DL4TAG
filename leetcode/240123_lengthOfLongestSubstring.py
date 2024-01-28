#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240123_lengthOfLongestSubstring.py
# Time       ：2024/1/23 10:10
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：3. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        我的思路：利用列表下标遍历字符串。
            如果当前字符不重复，则加入当前子字符串；
            如果当前字符已在子字符串，则获得当前字符前最右边的重复字符位置，从后一位重新开始维护子字符串。
        在子字符串有删减或者最终输出时更新longest_num，但这种方法无法得知最长无重复子字符串到底是哪个。
        推测在更新longes_num时记录此时下标即可输出最长无重复子字符串。
        """
        longest_num = 0
        curr_substr = []
        for c in range(len(s)):
            if s[c] not in curr_substr:
                curr_substr.append(s[c])
            else:
                longest_num = max(len(curr_substr), longest_num)
                # window_position = s.find("".join(curr_substr))
                rightmost_char_position = s.rfind(s[c], 0, c)
                curr_substr = list(s[rightmost_char_position+1: c+1])
            # print(s[c], '\t', curr_substr)
        longest_num = max(len(curr_substr), longest_num)
        return longest_num


s = Solution()
print(s.lengthOfLongestSubstring("bprkpqlbtqpqphr"))
