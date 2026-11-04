"""Recursive automata and tensor-based context-free path querying."""

from networkx import DiGraph
from pyformlang.cfg import CFG, Terminal
from pyformlang.finite_automaton import (
    Epsilon,
    EpsilonNFA,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Build one finite automaton for the productions of each variable."""
    boxes = set()
    variable_names = {
        variable.value for variable in cfg.variables | {cfg.start_symbol}
    }
    for variable in cfg.variables | {cfg.start_symbol}:
        automaton = EpsilonNFA()
        automaton.add_start_state(0)
        next_state = 1
        for production in cfg.productions:
            if production.head != variable:
                continue
            state = 0
            for symbol in production.body:
                # Preserve the distinction if a terminal shares a variable's name.
                value = symbol.value
                if isinstance(symbol, Terminal) and value in variable_names:
                    value = symbol
                automaton.add_transition(state, Symbol(value), next_state)
                state = next_state
                next_state += 1
            if state == 0:
                automaton.add_transition(0, Epsilon(), next_state)
                state = next_state
                next_state += 1
            automaton.add_final_state(state)
        boxes.add(Box(automaton, Symbol(variable.value)))
    return RecursiveAutomaton(initial_label=Symbol(cfg.start_symbol.value), boxes=boxes)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Build a recursive automaton from pyformlang's EBNF notation."""
    return RecursiveAutomaton.from_text(ebnf)


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
    combined = NondeterministicFiniteAutomaton()
    for label, box in rsm.boxes.items():
        for state in box.dfa.states:
            combined.add_start_state(State((label, state)))
        for source, symbol, target in box.dfa:
            value = symbol.value
            if isinstance(value, Terminal):
                product_symbol = Symbol(("terminal", value.value))
            elif symbol in rsm.labels:
                product_symbol = Symbol(("variable", value))
            else:
                product_symbol = Symbol(("terminal", value))
            combined.add_transition(
                State((label, source)), product_symbol, State((label, target))
            )
    machine = AdjacencyMatrixFA(combined)
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    graph_matrix.matrices = {
        Symbol(("terminal", symbol.value)): matrix
        for symbol, matrix in graph_matrix.matrices.items()
    }
    width = graph_matrix.states_count
    shape = (width, width)

    while True:
        product = intersect_automata(machine, graph_matrix)
        closure = product.transitive_closure()
        changed = False
        box_results = {}
        for label, box in rsm.boxes.items():
            rows, columns = [], []
            final_states = {
                machine.state_to_index[State((label, state))]
                for state in box.final_states
            }
            for state in box.start_state:
                machine_start = machine.state_to_index[State((label, state))]
                for source in range(width):
                    row = closure.getrow(machine_start * width + source)
                    for target in row.indices:
                        if target // width in final_states:
                            rows.append(source)
                            columns.append(target % width)
            result = csr_matrix(
                ([True] * len(rows), (rows, columns)), shape=shape, dtype=bool
            )
            box_results[label] = result
            variable_label = Symbol(("variable", label.value))
            previous = graph_matrix.matrices.get(
                variable_label, csr_matrix(shape, dtype=bool)
            )
            expanded = previous + result
            if expanded.nnz != previous.nnz:
                changed = True
            graph_matrix.matrices[variable_label] = expanded
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
