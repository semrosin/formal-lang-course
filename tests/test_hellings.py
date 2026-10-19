"""Context-free path queries, including epsilon and recursive grammars."""

from copy import deepcopy

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.hellings import hellings_based_cfpq


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_hellings_matches_balanced_paths_through_intermediate_vertices(
    graph_type,
) -> None:
    graph = graph_type()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")
    graph.add_edges_from([(30, 40), (40, 50), (50, 60)], label="b")
    cfg = CFG.from_text("S -> a S b | a b")

    assert hellings_based_cfpq(cfg, graph) == {(20, 40), (10, 50)}
    assert hellings_based_cfpq(cfg, graph, {10}, {50}) == {(10, 50)}


@pytest.mark.parametrize(
    "starts, finals, expected",
    [
        (None, None, {(10, 20), (20, 30)}),
        (set(), set(), {(10, 20), (20, 30)}),
        ({10}, None, {(10, 20)}),
        ({10}, set(), {(10, 20)}),
        (None, {30}, {(20, 30)}),
        (set(), {30}, {(20, 30)}),
        ({10}, {30}, set()),
        ({99}, {30}, set()),
    ],
)
def test_hellings_filters_start_and_final_vertices(starts, finals, expected) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(10, 20), (20, 30)], label="a")

    assert (
        hellings_based_cfpq(CFG.from_text("S -> a"), graph, starts, finals) == expected
    )


@pytest.mark.parametrize(
    "grammar",
    ["S -> $", "S -> A\nA -> B\nB -> $", "S -> A B\nA -> $\nB -> $"],
)
def test_hellings_includes_epsilon_paths_at_isolated_vertices(grammar: str) -> None:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([10, 20, 30])

    assert hellings_based_cfpq(CFG.from_text(grammar), graph) == {
        (10, 10),
        (20, 20),
        (30, 30),
    }
    assert hellings_based_cfpq(CFG.from_text(grammar), graph, {10, 20}, {20, 30}) == {
        (20, 20)
    }


def test_hellings_combines_nullable_variables_around_a_nonempty_path() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="b")
    cfg = CFG.from_text("S -> A B C\nA -> $\nB -> b\nC -> $")

    assert hellings_based_cfpq(cfg, graph) == {(0, 1)}


def test_hellings_handles_recursive_ambiguity_and_cycles() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(0, 1), (1, 0), (1, 2)], label="a")
    graph.add_node(3)
    cfg = CFG.from_text("S -> S S | a | $")

    assert hellings_based_cfpq(cfg, graph) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 3),
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


@pytest.mark.parametrize("grammar", ["S -> a", "S -> A\nA -> A", "S -> c"])
def test_hellings_returns_no_pairs_when_no_path_matches(grammar: str) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="b")

    assert hellings_based_cfpq(CFG.from_text(grammar), graph) == set()


@pytest.mark.parametrize("cfg", [CFG(), CFG.from_text("S -> a | $")])
def test_hellings_on_an_empty_graph(cfg: CFG) -> None:
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


def test_hellings_does_not_change_its_inputs() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    snapshot = deepcopy(graph)
    cfg = CFG.from_text("S -> a S | $")
    variables = cfg.variables.copy()
    productions = set(cfg.productions)
    starts, finals = {0}, {1}

    assert hellings_based_cfpq(cfg, graph, starts, finals) == {(0, 1)}
    assert nx.utils.graphs_equal(graph, snapshot)
    assert cfg.variables == variables
    assert set(cfg.productions) == productions
    assert starts == {0}
    assert finals == {1}
