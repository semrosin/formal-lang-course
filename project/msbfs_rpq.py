"""Multiple-source BFS-based regular path queries."""

from collections import deque

from networkx import MultiDiGraph

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa, regex_to_dfa

__all__ = ["ms_bfs_based_rpq"]


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Return all ``(start, final)`` pairs accepted by ``regex``.

    The algorithm works by intersecting the regex DFA with the graph NFA and
    then running a breadth-first search from each start node separately through
    the product automaton. Any product state that is both accepting for the
    regex and final for the chosen graph node produces one answer pair.
    """

    graph_nodes = set(graph.nodes)
    starts = (set(start_nodes) if start_nodes else graph_nodes) & graph_nodes
    finals = (set(final_nodes) if final_nodes else graph_nodes) & graph_nodes
    if not starts or not finals:
        return set()

    regex_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, starts, finals))
    product = intersect_automata(regex_matrix, graph_matrix)
    width = graph_matrix.states_count

    neighbors: list[set[int]] = [set() for _ in range(product.states_count)]
    for matrix in product.matrices.values():
        for source in range(product.states_count):
            neighbors[source].update(
                int(target)
                for target in matrix.indices[
                    matrix.indptr[source] : matrix.indptr[source + 1]
                ]
            )

    result: set[tuple[int, int]] = set()
    for start in starts:
        graph_start = graph_matrix.state_to_index[start]
        initial = {
            regex_start * width + graph_start
            for regex_start in regex_matrix.start_states
        }
        visited = set(initial)
        queue = deque(initial)

        while queue:
            state = queue.popleft()
            if state in product.final_states:
                graph_state = graph_matrix.index_to_state[state % width]
                node = getattr(graph_state, "value", graph_state)
                result.add((start, node))

            for target in neighbors[state] - visited:
                visited.add(target)
                queue.append(target)

    return result
