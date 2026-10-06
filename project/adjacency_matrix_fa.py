"""Sparse matrix representation of finite automata for regular path queries."""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_matrix, eye, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """An NFA represented by one sparse adjacency matrix per symbol.

    Args:
        automaton: Automaton to represent. If ``None``, an empty automaton
            is created; this is convenient when the fields are filled later.
    """

    def __init__(
        self, automaton: NondeterministicFiniteAutomaton | None = None
    ) -> None:
        if automaton is None:
            self.index_to_state = ()
            self.state_to_index = {}
            self.states_count = 0
            self.start_states = set()
            self.final_states = set()
            self.matrices = {}
            return

        self.index_to_state = tuple(automaton.states)
        self.state_to_index = {
            state: index for index, state in enumerate(self.index_to_state)
        }
        self.states_count = len(self.index_to_state)
        self.start_states = {
            self.state_to_index[state] for state in automaton.start_states
        }
        self.final_states = {
            self.state_to_index[state] for state in automaton.final_states
        }

        edges = defaultdict(lambda: ([], []))
        for source, symbol, target in automaton:
            sources, targets = edges[symbol]
            sources.append(self.state_to_index[source])
            targets.append(self.state_to_index[target])

        self.matrices = {
            symbol: csr_matrix(
                (
                    [True] * len(sources),
                    (sources, targets),
                ),  # (data, (row_ind, col_ind))
                shape=(self.states_count, self.states_count),
                dtype=bool,
            )
            for symbol, (sources, targets) in edges.items()
        }

    def adjacency_matrix(self) -> csr_matrix:
        """Return the boolean union of the per-symbol matrices."""

        adjacency = csr_matrix((self.states_count, self.states_count), dtype=bool)
        for matrix in self.matrices.values():
            adjacency = adjacency + matrix
        return adjacency

    def transitive_closure(self) -> csr_matrix:
        """Return the reflexive-transitive closure of the adjacency matrix."""

        reachable = eye(self.states_count, format="csr", dtype=bool)
        for matrix in self.matrices.values():
            reachable = reachable + matrix
        while True:
            expanded = reachable + reachable @ reachable
            if expanded.nnz == reachable.nnz:
                return expanded
            reachable = expanded

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Check whether the automaton accepts ``word``.

        The vector of current states is multiplied by the matrix of the next
        symbol on every step; the word is accepted when the final vector
        reaches at least one final state.
        """

        if not self.start_states or not self.final_states:
            return False

        current = np.zeros(self.states_count, dtype=bool)
        current[list(self.start_states)] = True
        for symbol in word:
            matrix = self.matrices.get(Symbol(symbol))
            if matrix is None:
                return False
            current = np.asarray(current @ matrix, dtype=bool)
            if not current.any():
                return False
        return bool(current[list(self.final_states)].any())

    def is_empty(self) -> bool:
        """Return whether no final state is reachable from a start state.

        Reachability is computed by multiplying the start vector by the
        transitive closure of the adjacency matrix, so paths of length zero
        are taken into account as well.
        """

        if not self.start_states or not self.final_states:
            return True

        starts = np.zeros(self.states_count, dtype=bool)
        starts[list(self.start_states)] = True
        reachable = np.asarray(starts @ self.transitive_closure(), dtype=bool)
        return not reachable[list(self.final_states)].any()


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Build the synchronous product using a Kronecker matrix per label."""

    product = AdjacencyMatrixFA()
    width = automaton2.states_count
    product.index_to_state = tuple(
        (first, second)
        for first in automaton1.index_to_state
        for second in automaton2.index_to_state
    )
    product.state_to_index = {
        state: index for index, state in enumerate(product.index_to_state)
    }
    product.states_count = automaton1.states_count * width
    # State (first, second) is stored at first * width + second, matching kron.
    product.start_states = {
        first * width + second
        for first in automaton1.start_states
        for second in automaton2.start_states
    }
    product.final_states = {
        first * width + second
        for first in automaton1.final_states
        for second in automaton2.final_states
    }
    product.matrices = {
        symbol: kron(matrix, automaton2.matrices[symbol], format="csr")
        for symbol, matrix in automaton1.matrices.items()
        if symbol in automaton2.matrices
    }
    return product


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """Find graph node pairs joined by a path accepted by ``regex``.

    The pairs are extracted from the transitive closure of the tensor product
    of the automata for ``regex`` and ``graph``: every closure row of a
    product start state gives the reachable product states, and every reached
    final state produces a pair of graph nodes.
    """

    regex_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    product = intersect_automata(regex_matrix, graph_matrix)
    closure = product.transitive_closure()

    result = set()
    for source in product.start_states:
        for target in closure.getrow(source).indices:
            if target in product.final_states:
                result.add(
                    (
                        product.index_to_state[source][1].value,
                        product.index_to_state[target][1].value,
                    )
                )
    return result
