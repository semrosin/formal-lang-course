"""Matrix CFPQ cases beyond the course's shared CFPQ tests."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.matrix_cfpq import matrix_based_cfpq


def test_matrix_cfpq_on_digraph_with_nonconsecutive_vertices() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")
    graph.add_edges_from([(30, 40), (40, 50), (50, 60)], label="b")
    cfg = CFG.from_text("S -> a S b | a b")

    assert matrix_based_cfpq(cfg, graph) == {(20, 40), (10, 50)}


@pytest.mark.parametrize("nodes", [None, set()])
def test_matrix_cfpq_uses_all_vertices_for_default_filters(nodes) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")

    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph, nodes, nodes) == {
        (10, 20),
        (20, 30),
    }


def test_matrix_cfpq_treats_explicit_absent_vertices_as_isolated_states() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    cfg = CFG.from_text("S -> A\nA -> a | $")

    assert matrix_based_cfpq(cfg, graph, {0, 99}, {1, 99}) == {(0, 1), (99, 99)}


def test_matrix_cfpq_handles_multicharacter_labels_and_a_custom_start_symbol() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="ignored")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(1, 2, label="right")
    cfg = CFG.from_text(
        "Query -> left right\nS -> ignored", start_symbol=Variable("Query")
    )

    assert matrix_based_cfpq(cfg, graph) == {(0, 2)}


@pytest.mark.parametrize("grammar", ["S -> a | $", "S -> a"])
def test_matrix_cfpq_on_an_empty_graph(grammar: str) -> None:
    assert matrix_based_cfpq(CFG.from_text(grammar), nx.MultiDiGraph()) == set()


def test_matrix_cfpq_on_an_empty_grammar() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    assert matrix_based_cfpq(CFG(start_symbol=Variable("S")), graph) == set()


def test_matrix_cfpq_does_not_lose_paths_when_many_derivations_coincide() -> None:
    graph = nx.MultiDiGraph()
    for middle in range(1, 257):
        graph.add_edge(0, middle, label="a")
        graph.add_edge(middle, 257, label="b")

    assert matrix_based_cfpq(CFG.from_text("S -> a b"), graph) == {(0, 257)}
