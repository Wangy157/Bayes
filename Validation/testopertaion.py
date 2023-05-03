#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   testopertaion.py    
@Contact :   605420005@qq.com
@Author  :   Wang
@Date    :   2023/4/27 15:37
------------      --------    -----------

"""
import pickle
from itertools import chain

import numpy as np
import sys

from pgmpy.factors.discrete import TabularCPD
from tabulate import tabulate

np.set_printoptions(threshold=sys.maxsize)
from Bayes import MicrobialPrediction
import pandas as pd

def get_cpd(mle,node, weighted=False):

    self = mle

    np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
    np.set_printoptions(linewidth=10000)  # 这个参数填的是横向多宽
    state_counts = self.state_counts(node, weighted=weighted)

    # if a column contains only `0`s (no states observed for some configuration
    # of parents' states) fill that column uniformly instead
    state_counts.values[:, (state_counts.values == 0).all(axis=0)] = 1

    parents = sorted(self.model.get_parents(node))
    parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
    node_cardinality = len(self.state_names[node])

    # Get the state names for the CPD
    state_names = {node: list(state_counts.index)}
    if parents:
        state_names.update(
            {
                state_counts.columns.names[i]: list(state_counts.columns.levels[i])
                for i in range(len(parents))
            }
        )

    cpd = TabularCPD(
        node,
        node_cardinality,
        np.array(state_counts),
        evidence=parents,
        evidence_card=parents_cardinalities,
        state_names={var: self.state_names[var] for var in chain([node], parents)},
    )


    f = open("CPD.txt","w")
    f.write(str(cpd))
    print(cpd)
    cpd.normalize()
    return cpd



pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.max_rows', None)
# MicrobialPrediction.getBayes("DBN_THSD.csv", 0.2)
from pgmpy.inference import VariableElimination


with open('estimated_model.pkl', 'rb') as f:
    model = pickle.load(f)

from pgmpy.estimators import MaximumLikelihoodEstimator

data = MicrobialPrediction.DBN_data_preprocess("DBN_THSD.csv", 0.2)

from pgmpy.models import BayesianModel


# 转成贝叶斯
model_struct = BayesianModel(ebunch=model.edges())


np.set_printoptions(threshold=10000) # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=10000) # 这个参数填的是横向多宽
# mle = MaximumLikelihoodEstimator(model=model_struct, data=data)

# 和训练数据拟合
model_struct.fit(data, estimator=MaximumLikelihoodEstimator)

# 获得CPD
mle = MaximumLikelihoodEstimator(model=model_struct, data=data)
# Estimating the CPD for a single node.
for node in model.nodes:
    if str(node).startswith("Pro"):
        predict = get_cpd(mle=mle, node=node)
        # result = mle.predict()
        np.set_printoptions(threshold=1e6)
        f.write(str(predict))
        print(predict)
        print(mle.estimate_cpd(node='CVP'))

# 测试
predict_data_raw = pd.read_csv("df_output_0.5_pre.csv")
predict_data = pd.DataFrame()
for d in predict_data_raw.columns:

    if d in model_struct.nodes:
        predict_data[d] = predict_data_raw[d]
# predict_data.set_index('id', inplace=True)
print(predict_data)
with open("01_pre_tmp.txt","w") as f:
    f.write(str(predict_data))
random = 0
# predict_data.replace('\n', '', regex=True, inplace=True)
predict_data = predict_data.copy()

# 设置缺失值进行预测
missing_data = ""
for data_a in predict_data.columns:
    random += 1
    if not str(data_a).startswith("Pre"):
        if random % 10 == 0:
            print(data_a)
            predict_data.drop(data_a, axis=1, inplace=True)
            missing_data = data_a
            break

# 进行预测
print(predict_data.index)
y_pred = model_struct.predict_probability(predict_data)
print(y_pred)
y_pred = model_struct.predict(predict_data, stochastic=False)
print(y_pred)




