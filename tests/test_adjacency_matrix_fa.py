"""Behavioral tests for the sparse automata and tensor RPQ implementation."""

from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.adjacency_matrix_fa import (
    AdjacencyMatrixFA,
    intersect_automata,
    tensor_based_rpq,
)
from project.task2 import regex_to_dfa


def test_adjacency_matrix_nfa_accepts_nondeterministic_paths() -> None:
    automaton = NondeterministicFiniteAutomaton()
    automaton.add_start_state("start")
    automaton.add_final_state("end")
    automaton.add_transition("start", "a", "dead")
    automaton.add_transition("start", "a", "middle")
    automaton.add_transition("middle", "b", "end")

    matrix_automaton = AdjacencyMatrixFA(automaton)

    assert matrix_automaton.accepts(iter(["a", "b"]))
    assert not matrix_automaton.accepts(["a"])
    assert not matrix_automaton.accepts(["a", "c"])


def test_is_empty_includes_zero_length_paths_and_ignores_dead_cycles() -> None:
    automaton = NondeterministicFiniteAutomaton()
    automaton.add_start_state(10)
    automaton.add_final_state(20)
    automaton.add_transition(10, "a", 10)

    assert AdjacencyMatrixFA(automaton).is_empty()

    automaton.add_final_state(10)
    assert not AdjacencyMatrixFA(automaton).is_empty()


def test_intersection_synchronizes_labels() -> None:
    first = AdjacencyMatrixFA(regex_to_dfa("a (b | c)"))
    second = AdjacencyMatrixFA(regex_to_dfa("a b"))

    intersection = intersect_automata(first, second)

    assert intersection.accepts(["a", "b"])
    assert not intersection.accepts(["a", "c"])
    assert not intersection.is_empty()


def test_tensor_rpq_tracks_each_start_node_separately() -> None:
    graph = MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(10, 40, label="b")

    assert tensor_based_rpq("a b", graph, {10, 20}, {40}) == {
        (10, 40),
        (20, 40),
    }
