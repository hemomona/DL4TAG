#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : install_test.py
# Time       ：2023/7/17 11:15
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.9.12
# Description：
"""
# import torch
# print(torch.cuda.is_available())  # Should be True. Output True
# print(torch.cuda.device_count())  # Should be > 0.  Output 1

from autogluon.tabular import TabularDataset, TabularPredictor

# a dataset from the cover story of Nature issue 7887: AI-guided intuition for math theorems.
# The goal is to predict a knot’s signature based on its properties.
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
print(train_data.head())

# pandas didn’t correctly recognize this data type as categorical, AutoGluon will fix this issue.
label = 'signature'
print(train_data[label].describe())

# specifying the label column name and then train on the dataset with TabularPredictor.fit().
# AutoGluon will recognize this is a multi-class classification task, perform automatic feature engineering,
# train multiple models, and then ensemble the models to create the final predictor.
predictor = TabularPredictor(label=label).fit(train_data)

# load test data and predict
test_data = TabularDataset(f'{data_url}test.csv')
y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

predictor.evaluate(test_data, silent=True)

predictor.leaderboard(test_data, silent=True)
