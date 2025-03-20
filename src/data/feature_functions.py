import numpy as np
import networkx as nx
from src.data.Data_gen_utils import EDGE_TYPE


def in_centrality_with_fixed_N(D):
    in_cent_feature = dict.fromkeys(D.nodes)
    for key in in_cent_feature.keys():
        in_cent_feature[key] = D.in_degree(key) / 6
    return in_cent_feature


def out_centrality_with_fixed_N(D):
    out_cent_feature = dict.fromkeys(D.nodes)
    for key in out_cent_feature.keys():
        out_cent_feature[key] = D.out_degree(key) / 6
    return out_cent_feature


def random_feature(D):
    rand_feature = dict.fromkeys(D.nodes)
    for key in rand_feature.keys():
        rand_feature[key] = np.random.rand()
    return rand_feature


def constant_feature(D):
    const_feature = dict.fromkeys(D.nodes)
    for key in const_feature.keys():
        const_feature[key] = 1
    return const_feature


def numbering_feature(D):
    num_feature = dict()
    for n, node in enumerate(D.nodes):
        num_feature[node] = n
    return num_feature


def get_sinks(D):
    return (node for node, out_dg in D.out_degree() if out_dg == 0)


def shortest_path_lengths(D):
    sinks = get_sinks(D)
    shortest_lengths = dict.fromkeys(D.nodes, float('inf'))
    for sink in sinks:
        for node, length in nx.shortest_path_length(D, target=sink).items():
            if shortest_lengths[node] > length:
                shortest_lengths[node] = length
    return shortest_lengths


def normalized_shortest_path_lengths(D):
    norm_short_feature = shortest_path_lengths(D)
    for key in norm_short_feature.keys():
        norm_short_feature[key] /= 6
    return norm_short_feature


def longest_path_length_to_target(D, target):
    dist = dict.fromkeys(D.nodes, -float('inf'))
    dist[target] = 0
    topo_order = reversed(list(nx.topological_sort(D)))
    for v in topo_order:
        for u in D.predecessors(v):
            if dist[u] < dist[v] + 1:
                dist[u] = dist[v] + 1
    return dist


def longest_path_lengths(D):
    sinks = get_sinks(D)
    longest_lengths = dict.fromkeys(D.nodes, -float('inf'))
    for sink in sinks:
        for node, length in longest_path_length_to_target(D, target=sink).items():
            if longest_lengths[node] < length:
                longest_lengths[node] = length
    return longest_lengths


def normalized_longest_path_lengths(D):
    norm_long_feature = longest_path_lengths(D)
    for key in norm_long_feature.keys():
        norm_long_feature[key] /= 6
    return norm_long_feature


def column_indicator(D):
    col_feature = dict()
    for node in D.nodes:
        col_feature[node] = -1
    for C in nx.weakly_connected_components(D):
        connected_D = D.subgraph(C)
        in_degs = dict()
        for node in C:
            in_degs[node] = [0, 0, 0]
        for u, v, wt in connected_D.edges.data('weight'):
            if wt == EDGE_TYPE.SINGLE_ARROW:
                in_degs[v][0] += 1
            elif wt == EDGE_TYPE.DASHED_ARROW:
                in_degs[v][1] += 1
            elif wt == EDGE_TYPE.DOUBLE_ARROW:
                in_degs[v][2] += 1

        first_row = [node for node in in_degs.keys() if in_degs[node][0] == 0]

        first_row_in_degs = dict()
        for node in first_row:
            first_row_in_degs[node] = [0, 0, 0]
            for v, _, wt in connected_D.in_edges(node, True):
                if not v in first_row: continue
                wt = wt['weight']
                if wt == EDGE_TYPE.SINGLE_ARROW:
                    first_row_in_degs[node][0] += 1
                elif wt == EDGE_TYPE.DASHED_ARROW:
                    first_row_in_degs[node][1] += 1
                elif wt == EDGE_TYPE.DOUBLE_ARROW:
                    first_row_in_degs[node][2] += 1

        candidates = []
        for node in first_row:
            if first_row_in_degs[node][1] == 0:
                candidates.append(node)
        if len(candidates) == 0:
            print("Something goes wrong!")
            return
        queue = [min(candidates)]
        for node in queue:
            candidates = []
            for _, v, wt in connected_D.out_edges(node, True):
                if not v in first_row: continue
                wt = wt['weight']
                if wt == EDGE_TYPE.DASHED_ARROW:
                    first_row_in_degs[v][1] -= 1
                elif wt == EDGE_TYPE.DOUBLE_ARROW:
                    first_row_in_degs[v][2] -= 1
            for v in first_row:
                if v in queue: continue
                if first_row_in_degs[v][1] == 0:
                    candidates.append(v)
            if candidates == []: break
            queue.append(min(candidates))
        for c in range(len(queue)):
            node = queue[c]
            col_queue = [node]
            for v in col_queue:
                col_feature[v] = c
                for _, u, wt in connected_D.out_edges(v, True):
                    wt = wt['weight']
                    if wt == EDGE_TYPE.SINGLE_ARROW:
                        in_degs[u][0] -= 1
                        if in_degs[u][0] == 0:
                            col_queue.append(u)
    return col_feature


def normalized_column_indicator(D):
    col_feature = dict()
    for node in D.nodes:
        col_feature[node] = -1
    for C in nx.weakly_connected_components(D):
        connected_D = D.subgraph(C)
        in_degs = dict()
        for node in C:
            in_degs[node] = [0, 0, 0]
        for u, v, wt in connected_D.edges.data('weight'):
            if wt == EDGE_TYPE.SINGLE_ARROW:
                in_degs[v][0] += 1
            elif wt == EDGE_TYPE.DASHED_ARROW:
                in_degs[v][1] += 1
            elif wt == EDGE_TYPE.DOUBLE_ARROW:
                in_degs[v][2] += 1

        first_row = [node for node in in_degs.keys() if in_degs[node][0] == 0]

        first_row_in_degs = dict()
        for node in first_row:
            first_row_in_degs[node] = [0, 0, 0]
            for v, _, wt in connected_D.in_edges(node, True):
                if not v in first_row: continue
                wt = wt['weight']
                if wt == EDGE_TYPE.SINGLE_ARROW:
                    first_row_in_degs[node][0] += 1
                elif wt == EDGE_TYPE.DASHED_ARROW:
                    first_row_in_degs[node][1] += 1
                elif wt == EDGE_TYPE.DOUBLE_ARROW:
                    first_row_in_degs[node][2] += 1

        candidates = []
        for node in first_row:
            if first_row_in_degs[node][1] == 0:
                candidates.append(node)
        if len(candidates) == 0:
            print("Something goes wrong!")
            return
        queue = [min(candidates)]
        for node in queue:
            candidates = []
            for _, v, wt in connected_D.out_edges(node, True):
                if not v in first_row: continue
                wt = wt['weight']
                if wt == EDGE_TYPE.DASHED_ARROW:
                    first_row_in_degs[v][1] -= 1
                elif wt == EDGE_TYPE.DOUBLE_ARROW:
                    first_row_in_degs[v][2] -= 1
            for v in first_row:
                if v in queue: continue
                if first_row_in_degs[v][1] == 0:
                    candidates.append(v)
            if candidates == []: break
            queue.append(min(candidates))
        for c in range(len(queue)):
            node = queue[c]
            col_queue = [node]
            for v in col_queue:
                col_feature[v] = float(c / len(queue))
                for _, u, wt in connected_D.out_edges(v, True):
                    wt = wt['weight']
                    if wt == EDGE_TYPE.SINGLE_ARROW:
                        in_degs[u][0] -= 1
                        if in_degs[u][0] == 0:
                            col_queue.append(u)
    return col_feature


def normalized_column_rev_indicator(D):
    col_feature = dict()
    for node in D.nodes:
        col_feature[node] = -1
    for C in nx.weakly_connected_components(D):
        connected_D = D.subgraph(C)
        in_degs = dict()
        for node in C:
            in_degs[node] = [0, 0, 0]
        for u, v, wt in connected_D.edges.data('weight'):
            if wt == EDGE_TYPE.SINGLE_ARROW:
                in_degs[v][0] += 1
            elif wt == EDGE_TYPE.DASHED_ARROW:
                in_degs[v][1] += 1
            elif wt == EDGE_TYPE.DOUBLE_ARROW:
                in_degs[v][2] += 1

        first_row = [node for node in in_degs.keys() if in_degs[node][0] == 0]

        first_row_in_degs = dict()
        for node in first_row:
            first_row_in_degs[node] = [0, 0, 0]
            for v, _, wt in connected_D.in_edges(node, True):
                if not v in first_row: continue
                wt = wt['weight']
                if wt == EDGE_TYPE.SINGLE_ARROW:
                    first_row_in_degs[node][0] += 1
                elif wt == EDGE_TYPE.DASHED_ARROW:
                    first_row_in_degs[node][1] += 1
                elif wt == EDGE_TYPE.DOUBLE_ARROW:
                    first_row_in_degs[node][2] += 1

        candidates = []
        for node in first_row:
            if first_row_in_degs[node][1] == 0:
                candidates.append(node)
        if len(candidates) == 0:
            print("Something goes wrong!")
            return
        queue = [min(candidates)]
        for node in queue:
            candidates = []
            for _, v, wt in connected_D.out_edges(node, True):
                if not v in first_row: continue
                wt = wt['weight']
                if wt == EDGE_TYPE.DASHED_ARROW:
                    first_row_in_degs[v][1] -= 1
                elif wt == EDGE_TYPE.DOUBLE_ARROW:
                    first_row_in_degs[v][2] -= 1
            for v in first_row:
                if v in queue: continue
                if first_row_in_degs[v][1] == 0:
                    candidates.append(v)
            if candidates == []: break
            queue.append(min(candidates))
        for c in range(len(queue)):
            node = queue[c]
            col_queue = [node]
            for v in col_queue:
                col_feature[v] = 1 - float(c / len(queue))
                for _, u, wt in connected_D.out_edges(v, True):
                    wt = wt['weight']
                    if wt == EDGE_TYPE.SINGLE_ARROW:
                        in_degs[u][0] -= 1
                        if in_degs[u][0] == 0:
                            col_queue.append(u)
    return col_feature
