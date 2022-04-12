from fileProcessing import FileProcessing
from MyGA import MyGA
import matplotlib
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import random
import pandas as pd

from timeit import default_timer as timer


def drawNetwork(G):
    """绘制带边标签的网络图"""
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=100, with_labels=True, font_size=5)
    # nx.draw_spring(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, alpha=0.9, rotate=True)
    # nx.draw(G, pos, with_labels=True)
    # nx.draw_spring(G, with_labels=True)
    plt.show()


def find_min(dual_list):
    """返回最短用时的出行方式下标"""
    if dual_list[0] == -1 and dual_list[1] == -1:
        return -1
    elif dual_list[0] != -1 and dual_list[1] != -1:
        return dual_list.index(min(dual_list))
    elif dual_list[0] != -1 and dual_list[1] == -1:
        return 0
    elif dual_list[0] == -1 and dual_list[1] != -1:
        return 1


def change_link(start, end, dual_graph, transport_index, min_price_graph, add_time=100):
    """手动增加出租车用时"""
    # min_price_graph[start][end] = 999
    # min_price_graph[end][start] = 999

    dual_graph[start][end][1] += add_time
    dual_graph[end][start] = dual_graph[start][end]

    transport_index[start][end] = find_min(dual_graph[start][end])
    transport_index[end][start] = transport_index[start][end]

    min_price_graph[start][end] = dual_graph[start][end][transport_index[start][end]]
    min_price_graph[end][start] = min_price_graph[start][end]
    pass


if __name__ == '__main__':

    start = timer()

    # 指定输入文件夹
    fp = FileProcessing("ga_input_data_7_9_test")
    # fp = FileProcessing("ga_input_data_0_24")
    # fp = FileProcessing("ga_input_data_7_9")
    # fp = FileProcessing("ga_input_data_16_19")

    # 加载邻接矩阵
    dual_graph = fp.load_data("Undirected_community_dual_matrix")
    transport_index = fp.load_data("transport_index")
    min_price_graph = fp.load_data("min_price_graph")

    # ================================= 手动修改数据 ============================================
    # change_link(286, 235, dual_graph, transport_index, min_price_graph)
    # change_link(286, 233, dual_graph, transport_index, min_price_graph)
    # change_link(291, 210, dual_graph, transport_index, min_price_graph, 200)
    # change_link(19, 4, dual_graph, transport_index, min_price_graph)
    # change_link(2, 4, dual_graph, transport_index, min_price_graph)
    # change_link(30, 4, dual_graph, transport_index, min_price_graph)
    # change_link(6, 4, dual_graph, transport_index, min_price_graph)
    # change_link(20, 2, dual_graph, transport_index, min_price_graph)
    # change_link(41, 4, dual_graph, transport_index, min_price_graph)
    # change_link(21, 2, dual_graph, transport_index, min_price_graph)
    # change_link(21, 18, dual_graph, transport_index, min_price_graph)
    # change_link(21, 0, dual_graph, transport_index, min_price_graph)
    # ================================= 手动修改数据 ============================================

    node_len = len(min_price_graph)
    input_nodes = []
    for i in range(node_len):
        if i != 90 and i != 96:
            input_nodes.append(str(i))

    input_edges = []
    count = 0
    for i in range(node_len):
        for j in range(node_len):
            if min_price_graph[i][j] != -1:
                input_edges.append((str(i), str(j), min_price_graph[i][j]))
    # print(input_edges)




    G = nx.Graph()
    # 往图添加节点和边
    G.add_nodes_from(input_nodes)
    G.add_weighted_edges_from(input_edges)
    # drawNetwork(G)
    # 296 286
    # "19", "221"
    ga = MyGA(G, min_price_graph, transport_index, input_nodes, ["161", "342"], 100, 40, 0.8, 0.05)
    x, y = ga.run()
    fp.save_file(ga.all_history_Y, "all_history_Y")

    # 绘图
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red', markersize=2)
    ax[0].set_title("all_history_Y")
    ax[1].plot(ga.generation_best_Y)
    ax[1].set_title("generation_best_Y")
    ax[2].plot(ga.generation_avg_Y)
    ax[2].set_title("generation_avg_Y")
    plt.show()

    end = timer()

    print("using time: ", end - start)
    pass


