"""Behavioral cases for GLL-based context-free reachability."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.tensor_cfpq import cfg_to_rsm


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
