#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : 240122_reverseList.py
# Time       ：2024/1/22 14:20
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
方法一迭代iterate：时间O(n)，空间O(1)
方法二递归recursion：时间O(n)，空间O(n)
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr is not None:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        # 先递归到最后一个节点，再返回最后一个节点
        newHead = reverseList(head.next)
        # 此时head.next.next = None，newHead = head.next末尾节点
        # 下一句实现了head.next指向head
        head.next.next = head
        head.next = None
        return newHead
