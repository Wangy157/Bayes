#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MicrobialPediction.py
@Contact :   605420005@qq.com
@Author  :   Wang
@Date    :   2023/4/20 10:17

"""
# ====================BN模型=========================
# 贝叶斯模型
import networkx as nx
from pgmpy.estimator import MaximumLikelihoodEstimator
from matplotlib import pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from sklearn.metrics import f1_score

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling


import pandas as pd
import numpy as np


# Funtion to evaluate the learned model structures.
def get_f1_score(estimated_model, true_model):
    nodes = estimated_model.nodes()
    est_adj = nx.to_numpy_matrix(estimated_model.to_undirected(), nodelist=nodes, weight=None)
    true_adj = nx.to_numpy_matrix(true_model.to_undirected(), nodelist=nodes, weight=None)

    f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score for the model skeleton: ", f1)

import pandas as pd

# 选择部分节点进行训练

# 读取 csv 文件并设置第一列为索引
df = pd.read_csv('THSD.csv', index_col=0)
df = df.sample(frac=1/15,random_state=42)
df.to_csv('sample_30.csv')
# 将第一列转换为数值型数据，并设置为索引
# df.index = pd.to_numeric(df.index.str.split('_').str[1])

# 选取从第二列开始到最后一列，按照每三列一组计算平均值
new_cols = [f'Time{i//3+1}' for i in range(1, len(df.columns)-1, 3)]
df_mean = pd.DataFrame()
for i in range(1, len(df.columns)-1, 3):
    if i+2 < len(df.columns)-1:
        mean_values = df.iloc[:, i:i+3].mean(axis=1)
    else:
        mean_values = df.iloc[:, i:len(df.columns)-1].mean(axis=1)
    df_mean[new_cols[i//3]] = mean_values

# 将结果输出为一个新的 data frame
# df_output = df_mean.T.reset_index().rename(columns={'index': 'id'})
# 算差值


# 计算每列与前一列的差值
df_diff = df_mean.diff(axis=1)

# 将数据框转置回来
# df_diff = df_diff.T.reset_index()

print(df_diff.iloc[:, 1:])
diff_df = df_diff.iloc[:, 1:].applymap(lambda x: 1 if int(x) > 0 else -1)
print(diff_df)
# 将列名改为要求的格式
new_cols = ['diff_' + str(i) + '_' + str(i-1) for i in range(2, 11)]
# new_cols.insert(0, 'id')
print(new_cols)
diff_df.columns = new_cols

df_output = diff_df.T


print(df_output)
#
# hc = HillClimbSearch(df_output)
# best_model = hc.estimate()
scoring_method = K2Score(data=df_output)
est = HillClimbSearch(data=df_output)
estimated_model = est.estimate(scoring_method=scoring_method, max_indegree=318, max_iter=int(500))

# 输出学习后的贝叶斯网络结构
print(estimated_model.edges())
G = nx.DiGraph(estimated_model.edges())
pos = nx.spring_layout(G, k=10 / np.sqrt(len(G.nodes())))  # 为节点生成随机位置

# 修改节点大小和节点间距
node_size = 300
node_gap = 0.3
nx.draw(G, pos, with_labels=True, node_size=node_size, width=1.0,
        node_color='skyblue', edge_color='gray', font_size=5,
        font_weight='bold', font_color='black',
        arrowsize=8, arrowstyle='->',
        alpha=0.9, linewidths=0,
        node_shape='o',
        # node_edgecolors='gray',
        edgecolors='gray',
        connectionstyle=f'arc3, rad={node_gap:.2f}')

plt.savefig("network_result.jpg", dpi=300)
plt.show()  # 显示图形

child_dependencies = {}
parent_dependecies = {}
for node in estimated_model.nodes:
    parent_dependecies[node] =  estimated_model.get_parents(node)
    child_dependencies[node] = estimated_model.get_children(node)
print("--------child--------------------")
print(child_dependencies)
print("---------parent--------------------")
print(parent_dependecies)

print("--------CPD--------------------------")
cpds = []
for node in estimated_model.nodes():
    parents = estimated_model.get_parents(node)
    num_parents = len(parents)
    cpd = TabularCPD(
        variable=node,
        variable_card=2,
        values=[[1/(2**num_parents)]*2**num_parents]*2,
        evidence=parents,
        evidence_card=[2]*num_parents
    )
    cpds.append(cpd)
print(cpds)

mle = MaximumLikelihoodEstimator(model = estimated_model)
