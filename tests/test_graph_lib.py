import pydot
import pytest

from project import graph_lib


def test_get_graph_info():
    number_of_vertices, number_of_edges, labels = graph_lib.get_graph_info(
        "generations"
    )

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


def test_save_two_cycles_graph(tmp_path):
    path = tmp_path / "two_cycles_graph.dot"

    graph = graph_lib.save_two_cycles_graph(3, 2, 0, ("a", "b"), path)

    expected_nodes = {0, 1, 2, 3, 4, 5}
    expected_edges = {
        (0, 1, "a"),
        (1, 2, "a"),
        (2, 3, "a"),
        (3, 0, "a"),
        (0, 4, "b"),
        (4, 5, "b"),
        (5, 0, "b"),
    }

    assert set(graph.nodes) == expected_nodes
    assert set(graph.edges(data="label")) == expected_edges

    (dot_graph,) = pydot.graph_from_dot_file(str(path))

    assert {int(node.get_name()) for node in dot_graph.get_nodes()} == expected_nodes
    assert {
        (edge.get_source(), edge.get_destination(), edge.get_attributes()["label"])
        for edge in dot_graph.get_edges()
    } == {(str(u), str(v), c) for u, v, c in expected_edges}
