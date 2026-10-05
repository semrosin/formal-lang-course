"""Sparse matrix representation of finite automata for regular path queries."""

from collections import defaultdict, deque
from collections.abc import Iterable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import csr_matrix, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """An NFA represented by one sparse adjacency matrix per symbol."""

    def __init__(self, automaton: NondeterministicFiniteAutomaton) -> None:
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

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Check whether any path labeled by ``word`` reaches a final state."""

        current = self.start_states
        for symbol in word:
            matrix = self.matrices.get(Symbol(symbol))
            if matrix is None:
                return False
            current = {
                int(target)
                for source in current
                for target in matrix.indices[
                    matrix.indptr[source] : matrix.indptr[source + 1]
                ]
            }
            if not current:
                return False
        return not current.isdisjoint(self.final_states)

    def is_empty(self) -> bool:
        """Return whether no final state is reachable from a start state."""

        visited = set(self.start_states)
        queue = deque(visited)
        while queue:
            source = queue.popleft()
            if source in self.final_states:
                return False
            for matrix in self.matrices.values():
                for target in matrix.indices[
                    matrix.indptr[source] : matrix.indptr[source + 1]
                ]:
                    target = int(target)
                    if target not in visited:
                        visited.add(target)
                        queue.append(target)
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Build the synchronous product using a Kronecker matrix per label."""

    product = object.__new__(AdjacencyMatrixFA)
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
    product.start_states = {
        first * width + second  # second < width (width == max(automaton2.states))
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
    """Find graph node pairs joined by a path accepted by ``regex``."""

    graph_nodes = set(graph.nodes)
    starts = (set(start_nodes) if start_nodes else graph_nodes) & graph_nodes
    finals = (set(final_nodes) if final_nodes else graph_nodes) & graph_nodes
    if not starts or not finals:
        return set()

    regex_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, starts, finals))
    product = intersect_automata(regex_matrix, graph_matrix)
    width = graph_matrix.states_count

    neighbors = [set() for _ in range(product.states_count)]
    for matrix in product.matrices.values():
        for source in range(product.states_count):
            neighbors[source].update(
                int(target)
                for target in matrix.indices[
                    matrix.indptr[source] : matrix.indptr[source + 1]
                ]
            )

    result = set()
    for start in starts:
        graph_start = graph_matrix.state_to_index[State(start)]
        initial = {
            regex_start * width + graph_start  # graph_start < width
            for regex_start in regex_matrix.start_states
        }
        visited = set(initial)
        queue = deque(initial)
        while queue:
            state = queue.popleft()
            if state in product.final_states:
                final = graph_matrix.index_to_state[state % width].value
                result.add((start, final))
            for target in neighbors[state] - visited:
                visited.add(target)
                queue.append(target)
    return result
