from collections import Counter

import networkx as nx
import pydot
import pytest

from project import graph_lib


def test_get_graph_info():
    info = graph_lib.get_graph_info("generations")
    number_of_vertices, number_of_edges, labels = info

    assert isinstance(info, graph_lib.GraphInfo)
    assert number_of_vertices == 129
    assert number_of_edges == 273
    assert labels == {
        "equivalentClass",
        "first",
        "hasChild",
        "hasParent",
        "hasSex",
        "hasSibling",
        "hasValue",
        "intersectionOf",
        "inverseOf",
        "onProperty",
        "oneOf",
        "range",
        "rest",
        "sameAs",
        "someValuesFrom",
        "type",
        "versionInfo",
    }


def test_get_graph_info_unknown_graph():
    with pytest.raises(FileNotFoundError):
        graph_lib.get_graph_info("no_such_graph")


def test_save_two_cycles_graph_round_trip(tmp_path):
    path = tmp_path / "two_cycles_graph.dot"

    graph = graph_lib.save_two_cycles_graph(3, 2, 42, ("x", "y"), path)

    expected_nodes = {1, 2, 3, 4, 5, 42}
    expected_edges = {
        (42, 1, "x"),
        (1, 2, "x"),
        (2, 3, "x"),
        (3, 42, "x"),
        (42, 4, "y"),
        (4, 5, "y"),
        (5, 42, "y"),
    }

    assert set(graph.nodes) == expected_nodes
    assert graph.number_of_edges() == len(expected_edges)
    assert set(graph.edges(data="label")) == expected_edges

    (dot_graph,) = pydot.graph_from_dot_file(str(path))

    assert dot_graph.get_type() == "digraph"
    assert {int(node.get_name()) for node in dot_graph.get_nodes()} == expected_nodes
    assert len(dot_graph.get_edges()) == len(expected_edges)
    assert {
        (edge.get_source(), edge.get_destination(), edge.get_attributes()["label"])
        for edge in dot_graph.get_edges()
    } == {(str(u), str(v), c) for u, v, c in expected_edges}


def test_save_two_cycles_graph_negative_size_preserves_file(tmp_path):
    path = tmp_path / "existing.dot"
    original_content = b"digraph { original; }\n"
    path.write_bytes(original_content)

    with pytest.raises(nx.NetworkXError):
        graph_lib.save_two_cycles_graph(-1, 2, 0, ("a", "b"), path)

    assert path.read_bytes() == original_content


def test_save_two_cycles_graph_parallel_edges_round_trip(tmp_path):
    path = str(tmp_path / "parallel_edges.dot")

    graph = graph_lib.save_two_cycles_graph(1, 1, 1, ("a", "b"), path)
    (restored,) = pydot.graph_from_dot_file(str(path))

    assert set(graph.nodes) == {1, 2}
    assert restored.get_type() == "digraph"
    assert not restored.get_strict()
    assert {node.get_name() for node in restored.get_nodes()} == {"1", "2"}
    expected_edges = Counter(
        {("1", "1", "a"): 2, ("1", "2", "b"): 1, ("2", "1", "b"): 1}
    )
    assert (
        Counter(
            (str(source), str(target), label)
            for source, target, label in graph.edges(data="label")
        )
        == expected_edges
    )
    assert (
        Counter(
            (
                edge.get_source(),
                edge.get_destination(),
                edge.get_attributes()["label"],
            )
            for edge in restored.get_edges()
        )
        == expected_edges
    )
