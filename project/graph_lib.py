import cfpq_data
import networkx as nx

def get_graph_info(graph_name: str) -> tuple[int, int, set[str]]:
    graph = cfpq_data.graph_from_csv(cfpq_data.download(graph_name))
    labels = set(nx.get_edge_attributes(graph, "label").values())
    return graph.number_of_nodes(), graph.number_of_edges(), labels
