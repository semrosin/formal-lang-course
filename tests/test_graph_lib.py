import json
from collections import Counter
from pathlib import Path

import networkx as nx
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


@pytest.mark.parametrize("graph_name", ["no_such_graph", "", " "])
def test_get_graph_info_unknown_graph(graph_name):
    with pytest.raises(FileNotFoundError):
        graph_lib.get_graph_info(graph_name)


def test_get_graph_info_missing_csv(tmp_path, monkeypatch):
    path = tmp_path / "missing.csv"
    monkeypatch.setattr(graph_lib.cfpq_data, "download", {"local": path}.__getitem__)

    with pytest.raises(FileNotFoundError):
        graph_lib.get_graph_info("local")


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


@pytest.mark.parametrize("n, m", [(-1, 2), (2, -1), (-1, -1)])
def test_save_two_cycles_graph_negative_size_preserves_file(n, m, tmp_path):
    path = tmp_path / "existing.dot"
    original_content = b"digraph { original; }\n"
    path.write_bytes(original_content)

    with pytest.raises(nx.NetworkXError):
        graph_lib.save_two_cycles_graph(n, m, 0, ("a", "b"), path)

    assert path.read_bytes() == original_content


@pytest.mark.parametrize("n, m", [(0, 2), (2, 0), (0, 0)])
def test_save_two_cycles_graph_empty_cycle(n, m, tmp_path):
    path = tmp_path / "graph.dot"

    # CFPQ Data requires a node in each cycle besides the common node.
    with pytest.raises(IndexError):
        graph_lib.save_two_cycles_graph(n, m, 0, ("a", "b"), path)

    assert not path.exists()


@pytest.mark.parametrize("labels", [(), ("a",)])
def test_save_two_cycles_graph_missing_labels(labels, tmp_path):
    path = tmp_path / "graph.dot"

    with pytest.raises(IndexError):
        graph_lib.save_two_cycles_graph(2, 3, 0, labels, path)

    assert not path.exists()


def test_save_two_cycles_graph_missing_parent(tmp_path):
    path = tmp_path / "missing" / "graph.dot"

    with pytest.raises(FileNotFoundError):
        graph_lib.save_two_cycles_graph(2, 3, 0, ("a", "b"), path)

    assert not path.parent.exists()


def test_save_two_cycles_graph_directory_path(tmp_path):
    with pytest.raises((IsADirectoryError, PermissionError)):
        graph_lib.save_two_cycles_graph(2, 3, 0, ("a", "b"), tmp_path)

    assert tmp_path.is_dir()
    assert not list(tmp_path.iterdir())


def _decode_dot_string(value):
    # The quoted identifiers and labels in these cases use JSON-compatible escapes.
    return json.loads(value) if value.startswith('"') else value


@pytest.mark.parametrize("path_type", [str, Path], ids=["str-path", "pathlib-path"])
@pytest.mark.parametrize(
    "n, m, common_node, labels, expected_counts",
    [
        pytest.param(1, 1, 0, ("a", "b"), (3, 4), id="smallest-cycles"),
        pytest.param(3, 2, 42, ("x", "y"), (6, 7), id="unequal-cycles"),
        pytest.param(2, 3, -7, ("a", "b"), (6, 7), id="negative-common-node"),
        pytest.param(2, 2, "center", ("a", "a"), (5, 6), id="same-labels"),
        pytest.param(
            2,
            1,
            "shared node",
            ("left label", "right:label"),
            (4, 5),
            id="quoted-identifiers-and-labels",
        ),
        pytest.param(1, 2, 0, ('a "quoted" label', "b"), (4, 5), id="quotes"),
        pytest.param(1, 1, 1, ("a", "b"), (2, 4), id="parallel-self-loops"),
    ],
)
def test_save_two_cycles_graph_round_trip(
    n, m, common_node, labels, expected_counts, path_type, tmp_path
):
    path = path_type(tmp_path / "round trip.dot")

    graph = graph_lib.save_two_cycles_graph(n, m, common_node, labels, path)
    (restored,) = pydot.graph_from_dot_file(str(path))

    assert (graph.number_of_nodes(), graph.number_of_edges()) == expected_counts
    assert restored.get_type() == "digraph"
    assert not restored.get_strict()
    assert {_decode_dot_string(node.get_name()) for node in restored.get_nodes()} == {
        str(node) for node in graph.nodes
    }
    restored_edges = Counter(
        (
            _decode_dot_string(edge.get_source()),
            _decode_dot_string(edge.get_destination()),
            _decode_dot_string(edge.get_attributes()["label"]),
        )
        for edge in restored.get_edges()
    )
    assert restored_edges == Counter(
        (str(source), str(target), label)
        for source, target, label in graph.edges(data="label")
    )
