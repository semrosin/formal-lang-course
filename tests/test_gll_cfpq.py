"""Behavioral cases for GLL-based context-free reachability."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm


def test_recursive_nullable_grammar_and_multiple_starts():
    from project.gll_cfpq import gll_based_cfpq

    graph = nx.MultiDiGraph()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")
    graph.add_edges_from([(30, 40), (40, 50)], label="b")
    graph.add_node(60)
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | $"))
    assert gll_based_cfpq(rsm, graph, {10, 20, 60}, {40, 50, 60}) == {
        (10, 50), (20, 40), (60, 60)
    }


def test_left_recursion_and_repeated_nullable_calls():
    from project.gll_cfpq import gll_based_cfpq

    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)], label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> S A | A\nA -> a | $"))
    assert gll_based_cfpq(rsm, graph, {0}, {0, 1, 2, 3}) == {
        (0, 0), (0, 1), (0, 2), (0, 3)
    }


def test_ebnf_parallel_edges_and_cyclic_graph():
    from project.gll_cfpq import gll_based_cfpq

    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="ignored")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(1, 0, label="right")
    assert gll_based_cfpq(ebnf_to_rsm("S -> (left right)*"), graph) == {
        (0, 0), (1, 1)
    }


@pytest.mark.parametrize("nodes", [None, set()])
def test_default_filters_include_all_vertices(nodes):
    from project.gll_cfpq import gll_based_cfpq

    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    assert gll_based_cfpq(cfg_to_rsm(CFG.from_text("S -> a")), graph, nodes, nodes) == {(0, 1)}


def test_explicit_missing_vertices_are_isolated():
    from project.gll_cfpq import gll_based_cfpq

    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> A\nA -> a | $"))
    assert gll_based_cfpq(rsm, graph, {0, 99}, {1, 99}) == {(0, 1), (99, 99)}
