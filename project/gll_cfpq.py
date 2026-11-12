"""GLL-based context-free reachability over recursive state machines."""

from collections import deque

from networkx import DiGraph
from pyformlang.rsa import RecursiveAutomaton

from project.task2 import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return pairs connected by words accepted by the initial RSM box.

    A descriptor records the current box and state, graph state, and GSS node.
    GSS edges record return states; completed calls are retained so that a
    return can also be propagated along edges discovered after its completion.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    descriptors = deque()
    seen = set()
    edges = {}
    popped = {}
    result = set()

    def add_descriptor(box_label, state, vertex, stack):
        descriptor = (box_label, state, vertex, stack)
        if descriptor not in seen:
            seen.add(descriptor)
            descriptors.append(descriptor)

    for vertex in graph_nfa.start_states:
        root = (None, vertex)
        for state in rsm.boxes[rsm.initial_label].start_state:
            add_descriptor(rsm.initial_label, state, vertex, root)

    while descriptors:
        label, state, vertex, stack = descriptors.popleft()
        box = rsm.boxes[label]

        for source, symbol, target in box.dfa:
            if source != state:
                continue
            if symbol in rsm.boxes:
                called_stack = (symbol, vertex)
                edge = (label, target, stack)
                outgoing = edges.setdefault(called_stack, set())
                if edge not in outgoing:
                    outgoing.add(edge)
                    for end in popped.get(called_stack, set()):
                        add_descriptor(label, target, end, stack)
                for start in rsm.boxes[symbol].start_state:
                    add_descriptor(symbol, start, vertex, called_stack)
            else:
                for graph_source, graph_symbol, graph_target in graph_nfa:
                    if graph_source == vertex and graph_symbol == symbol:
                        add_descriptor(label, target, graph_target, stack)

        if state in box.final_states:
            if stack[0] is None:
                if vertex in graph_nfa.final_states:
                    result.add((stack[1].value, vertex.value))
            else:
                completed = popped.setdefault(stack, set())
                if vertex not in completed:
                    completed.add(vertex)
                    for return_label, return_state, parent in edges.get(stack, set()):
                        add_descriptor(return_label, return_state, vertex, parent)

    return result
