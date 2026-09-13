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
