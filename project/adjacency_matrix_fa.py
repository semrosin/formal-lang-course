"""Sparse matrix representation of finite automata for regular path queries."""

from collections import defaultdict, deque
from collections.abc import Iterable

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_matrix


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
        for source, symbol, target in automaton._transition_function.get_edges():
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
