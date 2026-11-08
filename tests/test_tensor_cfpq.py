"""Cases for recursive-automaton conversion and tensor CFPQ."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import State, Symbol

from project.tensor_cfpq import _rsm_to_nfa, cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def test_combined_automaton_uses_only_box_start_states() -> None:
    rsm = cfg_to_rsm(CFG.from_text("S -> a b | $\nA -> c"))
    combined = _rsm_to_nfa(rsm)

    assert combined.start_states == {
        State((label, state))
        for label, box in rsm.boxes.items()
        for state in box.start_state
    }
    assert combined.states == {
        State((label, state))
        for label, box in rsm.boxes.items()
        for state in box.dfa.states
    }


def test_cfg_conversion_preserves_start_symbol_and_productions() -> None:
    cfg = CFG.from_text(
        "Query -> left Query right | $\nUnused -> other",
        start_symbol=Variable("Query"),
    )
    rsm = cfg_to_rsm(cfg)
    box = rsm.get_box(Symbol("Query"))

    assert rsm.initial_label == Symbol("Query")
    assert box.dfa.accepts([])
    assert box.dfa.accepts(["left", "Query", "right"])
    assert not box.dfa.accepts(["left", "right"])


def test_tensor_cfpq_handles_recursion_nullable_calls_and_node_filters() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(40, 50, label="b")
    graph.add_node(60)
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | $"))

    assert tensor_based_cfpq(rsm, graph, {10, 20, 60}, {40, 50, 60}) == {
        (10, 50),
        (20, 40),
        (60, 60),
    }


def test_tensor_cfpq_handles_ebnf_and_parallel_edges() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="ignored")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(1, 2, label="right")
    graph.add_edge(2, 3, label="left")
    graph.add_edge(3, 4, label="right")
    rsm = ebnf_to_rsm("S -> (left right)*")

    assert tensor_based_cfpq(rsm, graph, {0}, {2, 4}) == {(0, 2), (0, 4)}


@pytest.mark.parametrize("nodes", [None, set()])
def test_tensor_cfpq_default_filters_include_all_vertices(nodes) -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    assert tensor_based_cfpq(
        cfg_to_rsm(CFG.from_text("S -> a")), graph, nodes, nodes
    ) == {(0, 1)}


def test_tensor_cfpq_treats_explicit_absent_vertices_as_isolated() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> A\nA -> a | $"))

    assert tensor_based_cfpq(rsm, graph, {0, 99}, {1, 99}) == {(0, 1), (99, 99)}


def test_tensor_cfpq_with_empty_graph() -> None:
    graph = nx.DiGraph()
    assert tensor_based_cfpq(cfg_to_rsm(CFG.from_text("S -> $")), graph) == set()


def test_cfg_conversion_rejects_empty_grammar() -> None:
    with pytest.raises(ValueError):
        cfg_to_rsm(CFG(start_symbol=Variable("S")))
