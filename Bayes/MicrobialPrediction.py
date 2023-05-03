#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MicrobialPediction.py    
@Contact :   605420005@qq.com
@Author  :   Wang
@Date    :   2023/4/20 10:17
------------      --------    -----------

"""
import pickle

import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from sklearn.metrics import f1_score

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
import numpy as np

import pandas as pd

def DBN_data_preprocess(filename, sample_frac):
    """
    pre processing
    """

    df = pd.read_csv(filename, index_col=0)
    df = df.sample(frac=sample_frac, random_state=42)
    df.to_csv('sample'+str(sample_frac)+'.csv')
    # 将第一列转换为数值型数据，并设置为索引
    # df.index = pd.to_numeric(df.index.str.split('_').str[1])
    # 选取从第二列开始到最后一列，按照每三列一组计算平均值
    new_cols = [f'Time{i // 3 + 1}' for i in range(1, len(df.columns) - 1, 3)]
    # print(new_cols)
    df_mean = pd.DataFrame()
    new_cols = [str(df.columns[i]).split("_")[1] for i in range(1,len(df.columns)-1,3)]
    # print("cols:",new_cols)
    for i in range(1, len(df.columns) - 1, 3):
        if i + 2 < len(df.columns) - 1:
            mean_values = df.iloc[:, i:i + 3].mean(axis=1)
        else:
            mean_values = df.iloc[:, i:len(df.columns) - 1].mean(axis=1)
        df_mean[new_cols[i // 3]] = mean_values
    # 将结果输出为一个新的 data frame
    # df_output = df_mean.T.reset_index().rename(columns={'index': 'id'})
    # 算差值
    # 计算每列与前一列的差值
    # print("mean:")
    # print(df_mean)
    # new_cols.insert(0, 'id')
    # print(new_cols)
    df_mean.columns = new_cols
    df_output = df_mean.T
    # print(df_output)
    df_output.to_csv("df_output_"+str(sample_frac)+".csv")
    return df_output



def generate_blanklist(data):
    """
    generate whitelist
    """

    start_nodes = []
    end_nodes = []
    time_points = []
    for node in data.columns:
        # print(node)
        node = str(node)
        if not node.startswith('Pre'):
            # end_nodes.append(node)
            end_nodes.append(node)

        start_nodes.append(node)

    for node in data.T.columns:
        # print("time:",node)
        time_points.append(node)

    # edges = [(start, end, time) for start in start_nodes for end in end_nodes for time in time_points]
    edges = [(start, end) for start in start_nodes for end in end_nodes ]

    return edges


def DBN(target_data, whitelist):
    from pgmpy.estimators import HillClimbSearch
    scoring_method = K2Score(data=target_data)
    res= HillClimbSearch(target_data)
    estimated_model = res.estimate(white_list=whitelist, scoring_method=scoring_method, max_indegree=400, max_iter=int(10000))
    return estimated_model


def getBayes(filename, sampling_rate=1):
    data = DBN_data_preprocess(filename, sampling_rate)
    # data = pd.read_csv("df_output_0.06666666666666667.csv",index_col=0)
    whitelist = generate_blanklist(data)
    model = DBN(data, whitelist)
    plot_print_results(model)

def plot_print_results(estimated_model):
    """
    plot networkX figure and output results
    """
    G = nx.DiGraph(estimated_model.edges())
    pos = nx.spring_layout(G, k=10 / np.sqrt(len(G.nodes())))  # 为节点生成随机位置
    node_color = ['orange' if node.startswith('Pre') else 'skyblue' for node in G.nodes()]
    # 修改节点大小和节点间距
    node_size = 300
    node_gap = 0.3
    nx.draw(G, pos, with_labels=True, node_size=node_size, width=1.0,
            node_color=node_color, edge_color='gray', font_size=5,
            font_weight='bold', font_color='black',
            arrowsize=8, arrowstyle='->',
            alpha=0.9, linewidths=0,
            node_shape='o',
            # node_edgecolors='gray',
            edgecolors='gray',
            connectionstyle=f'arc3, rad={node_gap:.2f}')

    plt.savefig("network_result.jpg", dpi=300)
    plt.show()  # 显示图形

    with open('estimated_model.pkl', 'wb') as f:
        pickle.dump(estimated_model, f)
    output_parent_file = open("output_parent.txt","w")
    output_child_file = open("output_child.txt","w")
    output_CPD_file = open("output_CPD.txt","w")
    child_dependencies = {}
    parent_dependecies = {}
    for node in estimated_model.nodes:
        if str(node).startswith("Pro"):
            parent_dependecies[node] = estimated_model.get_parents(node)
            output_parent_file.write("child: "+str(node)+" parents: "+str(estimated_model.get_parents(node))+"\n")
        child_dependencies[node] = estimated_model.get_children(node)
        output_child_file.write("parent: "+str(node)+" child: "+str(estimated_model.get_children(node))+"\n")
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
            values=[[1 / (2 ** num_parents)] * 2 ** num_parents] * 2,
            evidence=parents,
            evidence_card=[2] * num_parents
        )
        cpds.append(cpd)
        output_CPD_file.write(str(cpd))
        print(cpd)
    output_CPD_file.close()
    output_child_file.close()
    output_parent_file.close()
    return estimated_model

