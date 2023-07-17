#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : protein_folding.py
# Time       ：2023/7/16 15:09
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
    This is a copy of protein_folding.ipynb from
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=949f360e

    It always clashes while preparing model

    I have not solve this problem
"""
# to set where pytorch saves its pretrained model
# import os
# os.environ['TORCH_HOME'] = 'D:/PycharmCache/torchcache'

# to be counted by google (can be skipped)
from transformers.utils import send_example_telemetry

send_example_telemetry("protein_folding_notebook", framework="pytorch")

# Now we load our model and tokenizer. If using GPU, use model.cuda() to transfer the model to GPU.
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
# tokenizer = AutoTokenizer.from_pretrained("D:/PycharmCache/torchcache/huggingface/hub/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a/")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.cuda()





