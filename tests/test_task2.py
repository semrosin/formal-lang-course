"""Tests for the Task 2 module: ``regex_to_dfa`` and ``graph_to_nfa``."""

import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project import graph_lib
from project.task2 import graph_to_nfa, regex_to_dfa

MINIMAL_DFA_REGRESSIONS = [
    "a",
    "a b",
    "a*",
    "a b c*",
    "(a | b)*c",
    "a*a*b",
    "(a | b | c)*(d | e | f)*",
    "((a | b)*c)*((d | e)*f)*",
    "(a b d) | (a b c)",
    "a b c*",
]


class TestRegexToDfa:
    @pytest.mark.parametrize("regex", MINIMAL_DFA_REGRESSIONS)
    def test_regex_to_dfa_is_deterministic_and_minimal(self, regex: str) -> None:
        dfa = regex_to_dfa(regex)
        minimized = dfa.minimize()

        assert dfa.is_deterministic()
        assert dfa.is_equivalent_to(minimized)
        assert len(minimized.states) == len(dfa.states)

    @pytest.mark.parametrize(
        ("regex", "accepted", "rejected"),
        [
            ("a", [["a"]], [[], ["b"], ["a", "a"]]),
            ("a b", [["a", "b"]], [["b", "a"], ["a"], ["b"]]),
            ("a*", [[], ["a"], ["a", "a"], ["a"] * 5], [["b"]]),
            ("a b c*", [["a", "b"], ["a", "b", "c"]], [["a"], ["b", "c"], ["a", "c"]]),
            (
                "(a | b)*c",
                [["c"], ["a", "c"], ["b", "b", "a", "c"]],
                [[], ["a"], ["c", "a"], ["a", "b"]],
            ),
            (
                "(a b) | (a c)",
                [["a", "b"], ["a", "c"]],
                [["a"], ["b"], ["a", "b", "c"]],
            ),
        ],
    )
    def test_regex_to_dfa_accepts_its_language(
        self, regex: str, accepted: list[list[str]], rejected: list[list[str]]
    ) -> None:
        dfa = regex_to_dfa(regex)

        for word in accepted:
            assert dfa.accepts(word), f"{regex} must accept {word}"

        for word in rejected:
            assert not dfa.accepts(word), f"{regex} must reject {word}"


def _two_cycles_graph() -> MultiDiGraph:
    """A graph with two cycles sharing the node ``c``: 0->1->c->0 and c->2->3->c."""
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, "c", label="a")
    graph.add_edge("c", 0, label="a")
    graph.add_edge("c", 2, label="b")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, "c", label="b")
    return graph


class TestGraphToNfa:
    def test_graph_to_nfa_accepts_words_from_graph(self) -> None:
        graph = MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="b")
        graph.add_edge(2, 3, label="c")
        graph.add_edge(3, 4, label="d")
        graph.add_edge(4, 5, label="e")

        nfa = graph_to_nfa(graph, {0}, {5})

        assert nfa.accepts(["a", "b", "c", "d", "e"])
        assert not nfa.accepts(["a", "b", "c", "d"])
        assert not nfa.accepts(["b", "c", "d", "e"])

    def test_graph_to_nfa_empty_start_and_final_sets_mean_all_nodes(self) -> None:
        graph = _two_cycles_graph()
        nfa = graph_to_nfa(graph.copy(), set(), set())

        assert {state.value for state in nfa.start_states} == set(graph.nodes)
        assert {state.value for state in nfa.final_states} == set(graph.nodes)

    def test_graph_to_nfa_explicit_start_and_final_states(self) -> None:
        graph = _two_cycles_graph()
        nfa = graph_to_nfa(graph.copy(), {0}, {0})

        assert nfa.accepts([])  # 0 is both start and final
        assert nfa.accepts(["a", "a", "a"])  # one full loop over the "a" cycle
        assert nfa.accepts(["a", "a", "a"] * 3)
        # 0 -> 1 -> c -> 2 -> 3 -> c -> 0, i.e. a lap over the "b" cycle
        assert nfa.accepts(["a", "a", "b", "b", "b", "a"])

        assert not nfa.accepts(["a"])
        assert not nfa.accepts(["a", "a"])
        assert not nfa.accepts(["b"])
        assert not nfa.accepts(["a", "a", "a", "b"])

    def test_graph_to_nfa_with_one_start_and_one_final_state(self) -> None:
        graph = _two_cycles_graph()
        nfa = graph_to_nfa(graph.copy(), {0}, {2})

        assert nfa.accepts(["a", "a", "b"])  # 0 -> 1 -> c -> 2
        assert nfa.accepts(["a", "a", "a", "a", "a", "b"])  # one lap over "a" first
        assert nfa.accepts(
            ["a", "a", "b", "b", "b", "a", "a", "a", "b"]
        )  # over "b" first

        assert not nfa.accepts([])
        assert not nfa.accepts(["a"])
        assert not nfa.accepts(["a", "a"])
        assert not nfa.accepts(["b"])

    def test_graph_to_nfa_keeps_parallel_edges(self) -> None:
        graph = MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(0, 1, label="b")

        nfa = graph_to_nfa(graph, {0}, {1})

        assert nfa.accepts(["a"])
        assert nfa.accepts(["b"])
        assert not nfa.accepts(["a", "b"])

    def test_graph_to_nfa_on_empty_graph(self) -> None:
        nfa = graph_to_nfa(MultiDiGraph(), set(), set())

        assert isinstance(nfa, NondeterministicFiniteAutomaton)
        assert nfa.is_empty()

    def test_graph_to_nfa_integration_with_task1_graph(self, tmp_path) -> None:
        path = tmp_path / "two_cycles.dot"
        graph = graph_lib.save_two_cycles_graph(
            3, 2, common_node=42, labels=("x", "y"), path=path
        )

        nfa = graph_to_nfa(graph, {42}, {42})

        # loop around the "x" cycle: 42 -> 1 -> 2 -> 3 -> 42
        assert nfa.accepts(["x", "x", "x", "x"])
        # loop around the "y" cycle: 42 -> 4 -> 5 -> 42
        assert nfa.accepts(["y", "y", "y"])
        # a mixed walk is not a valid path from 42 back to 42
        assert not nfa.accepts(["x", "x"])
        assert not nfa.accepts(["x", "y"])
