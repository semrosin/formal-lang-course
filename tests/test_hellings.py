"""Context-free path query cases not covered by the course autotests."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.hellings import hellings_based_cfpq


def test_hellings_on_digraph_with_nonconsecutive_vertices() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")
    graph.add_edges_from([(30, 40), (40, 50), (50, 60)], label="b")
    cfg = CFG.from_text("S -> a S b | a b")

    assert hellings_based_cfpq(cfg, graph) == {(20, 40), (10, 50)}
    assert hellings_based_cfpq(cfg, graph, {10}, {50}) == {(10, 50)}


@pytest.mark.parametrize("nodes", [None, set()])
def test_hellings_uses_all_vertices_for_default_filters(nodes) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")

    assert hellings_based_cfpq(CFG.from_text("S -> a"), graph, nodes, nodes) == {
        (10, 20),
        (20, 30),
    }


def test_hellings_handles_parallel_edges_and_multicharacter_labels() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="ignored")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(1, 2, label="right")
    cfg = CFG.from_text("S -> left right")

    assert hellings_based_cfpq(cfg, graph) == {(0, 2)}


def test_hellings_uses_the_grammar_start_symbol() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    cfg = CFG.from_text("Query -> a\nS -> b", start_symbol=Variable("Query"))

    assert hellings_based_cfpq(cfg, graph) == {(0, 1)}


def test_hellings_on_an_empty_graph() -> None:
    cfg = CFG.from_text("S -> a | $")
    assert hellings_based_cfpq(cfg, nx.MultiDiGraph()) == set()


def test_hellings_on_an_empty_grammar() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    assert hellings_based_cfpq(CFG(start_symbol=Variable("S")), graph) == set()


def test_hellings_treats_explicit_absent_vertices_as_isolated_states() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    cfg = CFG.from_text("S -> a | $")

    assert hellings_based_cfpq(cfg, graph, {0, 99}, {1, 99}) == {(0, 1), (99, 99)}
