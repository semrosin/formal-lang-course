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

    graph_nodes = set(graph.nodes)
    starts = (set(start_nodes) if start_nodes else graph_nodes) & graph_nodes
    finals = (set(final_nodes) if final_nodes else graph_nodes) & graph_nodes
    if not starts or not finals:
        return set()

    regex_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, starts, finals))

    width = regex_matrix.states_count
    sources = list(starts)
    shape = (len(sources) * width, graph_matrix.states_count)

    # Frontier and visited states: one row block per start node.
    front = np.zeros(shape, dtype=bool)
    for block, source in enumerate(sources):
        column = graph_matrix.state_to_index[source]
        for dfa_start in regex_matrix.start_states:
            front[block * width + dfa_start, column] = True
    visited = front.copy()

    dfa_steps = {
        symbol: block_diag([matrix.transpose()] * len(sources), format="csr")
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
    for block, source in enumerate(sources):
        block_rows = visited[block * width : (block + 1) * width, :]
        reached = np.any(block_rows[dfa_finals, :], axis=0)
        for final in finals:
            if reached[graph_matrix.state_to_index[final]]:
                result.add((source, final))

    return result
