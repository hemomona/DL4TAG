#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240128_myAtoi.py
# Time       ：2024/1/28 13:26
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：8. 字符串转换整数 (atoi)
请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
函数 myAtoi(string s) 的算法如下：
读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：
本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
"""


class Solution:
    """
    暴力解法，可读性差
    """

    def myAtoi(self, s: str) -> int:
        num_start = False
        neg_num = False
        num_str = ''
        for i in range(len(s)):
            if num_start:
                if '0' <= s[i] <= '9':
                    num_str += s[i]
                else:
                    break

            if not num_start:
                if s[i] == '-':
                    neg_num = True
                    num_start = True
                elif s[i] == '+':
                    num_start = True
                elif '0' <= s[i] <= '9':
                    num_str += s[i]
                    num_start = True
                elif not s[i] == ' ':
                    break

        num = int(num_str) if num_str else 0
        num = 0 - num if neg_num else num
        num = num if num <= 2 ** 31 - 1 else 2 ** 31 - 1
        num = num if num >= -2 ** 31 else -2 ** 31
        return num


if -1:
    print("-1 is True")

"""
确定有限状态机（deterministic finite automaton, DFA）:
程序在每个时刻有一个状态 s，每次从序列中输入一个字符 c，并根据字符 c 转移到下一个状态 s'。
状态s有start, signed, in_number, end
字符c有' ', '+'/'-', '0'-'9', others
那么可列出表格：
s\c         ' '     '+'/'-'     '0'-'9'     others
start       start   signed      in_numbrt   end
signed      end     end         in_number   end
in_number   end     end         in_number   end
end         end     end         end         end
"""
INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31


class Automaton:
    def __init__(self):
        self.state = 'start'
        self.sign = 1
        self.ans = 0
        self.table = {
            'start': ['start', 'signed', 'in_number', 'end'],
            'signed': ['end', 'end', 'in_number', 'end'],
            'in_number': ['end', 'end', 'in_number', 'end'],
            'end': ['end', 'end', 'end', 'end'],
        }

    def get_col(self, c):
        if c.isspace():
            return 0
        if c == '+' or c == '-':
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c):
        # 更新当前状态
        self.state = self.table[self.state][self.get_col(c)]

        if self.state == 'in_number':
            self.ans = self.ans * 10 + int(c)
            # ans只记录纯数字，因此下面为-INT_MIN；整数0为False，其它都为True
            self.ans = min(self.ans, INT_MAX) if self.sign == 1 else min(self.ans, -INT_MIN)
        elif self.state == 'signed':
            self.sign = 1 if c == '+' else -1


class Solution:
    def myAtoi(self, str: str) -> int:
        automaton = Automaton()
        for c in str:
            automaton.get(c)
        return automaton.sign * automaton.ans
