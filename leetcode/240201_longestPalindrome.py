#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240201_longestPalindrome.py
# Time       ：2024/2/1 9:42
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：5. 最长回文子串
给你一个字符串 s，找到 s 中最长的回文子串。
如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
"""


class Solution:
    """
    对于一个子串而言，如果它是回文串，并且长度大于 222，那么将它首尾的两个字母去除之后，它仍然是个回文串。
    例如对于字符串 “ababa” ，如果我们已经知道 “bab” 是回文串，那么 “ababa” 一定是回文串，这是因为它的首尾两个字母都是 “a”
    根据这样的思路，我们就可以用动态规划的方法解决本题。我们用 P(i,j) 表示字符串 s 的第 i 到 j 个字母组成的串（下文表示成 s[i, j]）是否为回文串：
    P(i,j) = True if s[i, j] is Palindrome; False if (s[i, j] is not palindrome) or i > j
    状态转移方程 P(i,j) = P(i+1,j−1)∧(Si==Sj)
    边界条件 P(i,i) = True; P(i,i+1) = whether Si==Si+1
    时间O(N2)，空间O(N2)
    """
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s

        max_len = 1
        begin = 0
        # dp[i][j]表示s[i][j]是否为回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True

        # L为子串长度，枚举【2，n】
        for L in range(2, n + 1):
            # i为左边界，枚举【0，n-1】
            for i in range(n):
                j = L + i - 1
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                # P(i,j)=P(i+1,j−1)∧(Si==Sj)
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]

                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]


class Solution:
    """
    状态转移链：P(i,j)←P(i+1,j−1)←P(i+2,j−2)←⋯←某一边界情况。可以发现，所有的状态在转移的时候的可能性都是唯一的。
    我们枚举所有的「回文中心」并尝试「扩展」，直到无法扩展为止，此时的回文串长度即为此「回文中心」下的最长回文串长度。我们对所有的长度求出最大值，即可得到最终的答案。
    时间O(N2)，空间O(1)
    """
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right -1

    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i+1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start:end+1]

