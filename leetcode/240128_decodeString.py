#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240128_decodeString.py
# Time       ：2024/1/28 20:41
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：394. 字符串解码
给定一个经过编码的字符串，返回它解码后的字符串。
编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
"""


class Solution:
    """
    我的解法，时间复杂度O(N2)
    """
    def decodeString(self, s: str) -> str:
        # 未找到返回-1，等价于True
        rightmost_openbracket = s.rfind('[')
        if not rightmost_openbracket:
            return s

        while rightmost_openbracket >= 0:
            leftmost_closebracket = s.find(']', rightmost_openbracket)
            repeat = s[rightmost_openbracket+1:leftmost_closebracket]

            n_str = ''
            n_index = rightmost_openbracket - 1
            while n_index >= 0:
                if '0' <= s[n_index] <= '9':
                    n_str = s[n_index] + n_str
                    n_index -= 1
                else:
                    break
            s = s[:n_index+1] + repeat*int(n_str) + s[leftmost_closebracket+1:]
            rightmost_openbracket = s.rfind('[')
        return s


s = Solution()
print(s.decodeString("100[leetcode]"))


class Solution:
    def decodeString(self, s: str) -> str:
        """
        辅助栈，先进后出
        """
        stack, res, multi = [], "", 0
        for c in s:
            if c == '[':
                # multi是[后面的重复次数，res是[前面的字符串
                stack.append([multi, res])
                res, multi = "", 0
            elif c == ']':
                # 遇到]后，res为这次[]内字符串，stack顶为这次[]前重复次数-再前一次[到这次[]间的字符
                curr_multi, last_res = stack.pop()
                res = last_res + curr_multi * res
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c)
            else:
                res += c
        return res


class Solution:
    def decodeString(self, s: str) -> str:
        """
        递归法，深度优先遍历
        """
        def dfs(s, i):
            res, multi = "", 0
            while i < len(s):
                if '0' <= s[i] <= '9':
                    multi = multi*10+int(s[i])
                elif s[i] == '[':
                    i, tmp = dfs(s, i+1)
                    res += multi*tmp
                    multi = 0
                elif s[i] == ']':
                    return i, res
                else:
                    res += s[i]
                i += 1
            return res
        return dfs(s, 0)
