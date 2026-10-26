"""Context-free path querying by sparse Boolean matrix multiplication."""

from networkx import DiGraph
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_matrix, eye

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.task2 import graph_to_nfa
from project.weak_normal_form import cfg_to_weak_normal_form

__all__ = ["matrix_based_cfpq"]


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return pairs connected by a path in the language of ``cfg``.

    Use Azimov and Grigorev's matrix algorithm (doi:10.1145/3210259.3210264)
    with one sparse Boolean CSR matrix per nonterminal. For each rule
    ``A -> B C``, add ``M_B @ M_C`` to ``M_A`` until no matrix changes.
    Epsilon rules initialize the corresponding matrices' diagonals.

    The grammar is converted to weak Chomsky normal form. Directed graphs
    and multigraphs with ``label`` edge attributes are supported. As in
    ``graph_to_nfa``, omitted or empty filters select all graph vertices;
    explicit vertices absent from the graph are treated as isolated states.
    Input objects are not modified.
    """
    normalized = cfg_to_weak_normal_form(cfg)
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    shape = (graph_matrix.states_count, graph_matrix.states_count)
    matrices = {
        variable: csr_matrix(shape, dtype=bool)
        for variable in normalized.variables | {normalized.start_symbol}
    }

    binary_rules = []
    for production in normalized.productions:
        head, body = production.head, production.body
        if len(body) == 2:
            binary_rules.append((head, body[0], body[1]))
        elif len(body) == 1:
            terminal_matrix = graph_matrix.matrices.get(Symbol(body[0].value))
            if terminal_matrix is not None:
                matrices[head] = matrices[head] + terminal_matrix
        else:
            matrices[head] = matrices[head] + eye(shape[0], format="csr", dtype=bool)

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_rules:
            expanded = matrices[head] + matrices[left] @ matrices[right]
            if expanded.nnz != matrices[head].nnz:
                changed = True
            matrices[head] = expanded

    reachable = matrices[normalized.start_symbol]
    return {
        (
            graph_matrix.index_to_state[source].value,
            graph_matrix.index_to_state[target].value,
        )
        for source in graph_matrix.start_states
        for target in reachable.getrow(source).indices
        if target in graph_matrix.final_states
    }
