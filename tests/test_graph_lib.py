import pydot
import pytest

from project import graph_lib


@pytest.mark.parametrize(
    "csv_data, expected",
    [
        pytest.param("", (0, 0, set()), id="empty-graph"),
        pytest.param(
            "0 1 a\n0 1 a\n0 1 b\n1 1 b\n2 0 a\n",
            (3, 5, {"a", "b"}),
            id="parallel-edges-and-self-loop",
        ),
    ],
)
def test_get_graph_info_named_fields(csv_data, expected, tmp_path, monkeypatch):
    path = tmp_path / "graph.csv"
    path.write_text(csv_data, encoding="utf-8")
    monkeypatch.setattr(graph_lib.cfpq_data, "download", {"local": path}.__getitem__)

    info = graph_lib.get_graph_info("local")

    assert isinstance(info, graph_lib.GraphInfo)
    assert info.number_of_vertices == expected[0]
    assert info.number_of_edges == expected[1]
    assert info.labels == expected[2]
    number_of_vertices, number_of_edges, labels = info
    assert (number_of_vertices, number_of_edges, labels) == expected


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
