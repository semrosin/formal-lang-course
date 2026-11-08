"""Recursive automata and tensor-based context-free path querying."""

from networkx import DiGraph
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Convert a CFG using pyformlang's text representation."""
    return RecursiveAutomaton.from_text(
        cfg.to_text(), start_symbol=Symbol(cfg.start_symbol.value)
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Build a recursive automaton from pyformlang's EBNF notation."""
    return RecursiveAutomaton.from_text(ebnf)


def _rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    """Combine the RSM's boxes into one finite automaton."""
    combined = NondeterministicFiniteAutomaton()
    for label, box in rsm.boxes.items():
        for state in box.start_state:
            combined.add_start_state(State((label, state)))
        for source, symbol, target in box.dfa:
            combined.add_transition(
                State((label, source)), symbol, State((label, target))
            )
    return combined


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Find graph vertex pairs connected by the language of the initial box.

    Repeatedly intersect the RSM with the graph, compute product reachability,
    and add an edge for each path accepted by a box until no edges are added.
    """
    machine = AdjacencyMatrixFA(_rsm_to_nfa(rsm))
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    width = graph_matrix.states_count
    shape = (width, width)

    while True:
        product = intersect_automata(machine, graph_matrix)
        closure = product.transitive_closure()
        changed = False
        box_results = {}
        for label, box in rsm.boxes.items():
            result = csr_matrix(shape, dtype=bool)
            final_states = {
                machine.state_to_index[State((label, state))]
                for state in box.final_states
            }
            for state in box.start_state:
                machine_start = machine.state_to_index[State((label, state))]
                for machine_final in final_states:
                    result = (
                        result
                        + closure[
                            machine_start * width : (machine_start + 1) * width,
                            machine_final * width : (machine_final + 1) * width,
                        ]
                    )
            box_results[label] = result
            previous = graph_matrix.matrices.get(label, csr_matrix(shape, dtype=bool))
            expanded = previous + result
            if expanded.nnz != previous.nnz:
                changed = True
            graph_matrix.matrices[label] = expanded
        if not changed:
            break

    result = box_results.get(rsm.initial_label, csr_matrix(shape, dtype=bool))
    return {
        (
            graph_matrix.index_to_state[source].value,
            graph_matrix.index_to_state[target].value,
        )
        for source in graph_matrix.start_states
        for target in result.getrow(source).indices
        if target in graph_matrix.final_states
    }
