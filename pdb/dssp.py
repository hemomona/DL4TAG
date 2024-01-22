#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : dssp.py
# Time       ：2023/10/27 1:20
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
"""
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
p = PDBParser()
structure = p.get_structure("6f9n", "./6f9n.pdb")
model = structure[0]
dssp = DSSP(model, "./6f9n.pdb", dssp='mkdssp')

