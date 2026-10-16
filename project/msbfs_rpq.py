"""Multiple-source BFS-based regular path queries."""

from networkx import MultiDiGraph
from scipy.sparse import block_diag, csr_matrix

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.task2 import graph_to_nfa, regex_to_dfa

__all__: list[str] = ["ms_bfs_based_rpq"]


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Return all ``(start, final)`` pairs accepted by ``regex``.

    A multiple-source BFS advances all start nodes simultaneously.
    Every BFS layer is computed with sparse Boolean matrix products:
    for a symbol the frontier is shifted along the graph edges with
    that label and, inside each block, advanced by the DFA transition matrix
    of the symbol (enlarged to all blocks at once with ``block_diag``).
    """

    regex_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    width: int = regex_matrix.states_count
    start_indices: list[int] = sorted(graph_matrix.start_states)
    blocks: int = len(start_indices)
    if not blocks:
        return set()

    row_indices: list[int] = []
    column_indices: list[int] = []
    for block, start in enumerate(start_indices):
        for dfa_start in regex_matrix.start_states:
            row_indices.append(block * width + dfa_start)
            column_indices.append(start)
    front = csr_matrix(
        ([True] * len(row_indices), (row_indices, column_indices)),
        shape=(blocks * width, graph_matrix.states_count),
        dtype=bool,
    )
    visited = front.copy()

    dfa_steps = {
        symbol: block_diag([matrix.transpose()] * blocks, format="csr")
        for symbol, matrix in regex_matrix.matrices.items()
        if symbol in graph_matrix.matrices
    }

    while front.nnz:
        new_front = None
        for symbol, dfa_step in dfa_steps.items():
            # Shift along the graph edges, then advance the DFA in each block.
            moved = dfa_step @ (front @ graph_matrix.matrices[symbol])
            new_front = moved if new_front is None else new_front + moved
        if new_front is None:
            break
        new_front = new_front.tocsr()
        # Keep only the states that have not been reached before.
        front = new_front - new_front.multiply(visited)
        front.eliminate_zeros()
        visited = visited + new_front

    if not regex_matrix.final_states:
        return set()

    selector_rows: list[int] = []
    selector_columns: list[int] = []
    for block in range(blocks):
        for dfa_final in regex_matrix.final_states:
            selector_rows.append(block)
            selector_columns.append(block * width + dfa_final)
    selector = csr_matrix(
        ([True] * len(selector_rows), (selector_rows, selector_columns)),
        shape=(blocks, blocks * width),
        dtype=bool,
    )
    reached = selector @ visited

    result: set[tuple[int, int]] = set()
    for block, start in enumerate(start_indices):
        start_value = graph_matrix.index_to_state[start].value
        for node in reached.getrow(block).indices:
            if node in graph_matrix.final_states:
                result.add((start_value, graph_matrix.index_to_state[node].value))

    return result
