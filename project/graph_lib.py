from typing import Any

import cfpq_data
import networkx as nx
import pydot
import pathlib


def get_graph_info(graph_name: str) -> tuple[int, int, set[str]]:
    graph = cfpq_data.graph_from_csv(cfpq_data.download(graph_name))
    labels = set(nx.get_edge_attributes(graph, "label").values())
    return graph.number_of_nodes(), graph.number_of_edges(), labels


def save_two_cycles_graph(
    n: int,
    m: int,
    common_node: int | Any,
    labels: tuple[str, str],
    path: str | pathlib.Path,
) -> nx.MultiDiGraph:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, common_node=common_node, labels=labels)

    dot_graph = pydot.Dot(graph_type="digraph")
    for node in graph.nodes:
        dot_graph.add_node(pydot.Node(str(node)))
    for source, target, label in graph.edges(data="label"):
        dot_graph.add_edge(pydot.Edge(str(source), str(target), label=label))

    dot_graph.write_raw(str(path))
    return graph
