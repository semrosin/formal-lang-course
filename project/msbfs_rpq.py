"""Multiple-source BFS-based regular path queries."""

import numpy as np
from networkx import MultiDiGraph
from scipy.sparse import block_diag

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.task2 import graph_to_nfa, regex_to_dfa

__all__ = ["ms_bfs_based_rpq"]


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Return all ``(start, final)`` pairs accepted by ``regex``.

    A multiple-source BFS advances all start nodes simultaneously.
    Every BFS layer is computed with Boolean matrix products:
    for a symbol the frontier is shifted along the graph edges with
    that label and, inside each block, advanced by the DFA transition matrix
    of the symbol (enlarged to all blocks at once with ``block_diag``).
    """

    regex_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    width = regex_matrix.states_count
    start_indices = sorted(graph_matrix.start_states)
    blocks = len(start_indices)
    if not blocks:
        return set()

    shape = (blocks * width, graph_matrix.states_count)

    # Frontier and visited states: one row block per start node.
    front = np.zeros(shape, dtype=bool)
    for block, start in enumerate(start_indices):
        for dfa_start in regex_matrix.start_states:
            front[block * width + dfa_start, start] = True
    visited = front.copy()

    dfa_steps = {
        symbol: block_diag([matrix.transpose()] * blocks, format="csr")
        for symbol, matrix in regex_matrix.matrices.items()
        if symbol in graph_matrix.matrices
    }

    while front.any():
        new_front = np.zeros(shape, dtype=bool)
        for symbol, dfa_step in dfa_steps.items():
            # Shift along the graph edges, then advance the DFA in each block.
            moved = dfa_step @ (front @ graph_matrix.matrices[symbol])
            new_front |= np.asarray(moved, dtype=bool)
        front = new_front & ~visited
        visited |= new_front

    dfa_finals = list(regex_matrix.final_states)
    result: set[tuple[int, int]] = set()
    for block, start in enumerate(start_indices):
        block_rows = visited[block * width : (block + 1) * width, :]
        reached = np.any(block_rows[dfa_finals, :], axis=0)
        for final in graph_matrix.final_states:
            if reached[final]:
                result.add(
                    (
                        graph_matrix.index_to_state[start].value,
                        graph_matrix.index_to_state[final].value,
                    )
                )

    return result
