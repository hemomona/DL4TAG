#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : protein_folding.py
# Time       ：2023/7/16 15:09
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.7
# Description：
"""

# to be counted by google (can be skipped)
from transformers.utils import send_example_telemetry

send_example_telemetry("protein_folding_notebook", framework="pytorch")

# Now we load our model and tokenizer. If using GPU, use model.cuda() to transfer the model to GPU.
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.cuda()





