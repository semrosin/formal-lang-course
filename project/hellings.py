"""Hellings' algorithm for context-free path queries."""

from networkx import DiGraph
from pyformlang.cfg import CFG, Terminal

from project.task2 import graph_to_nfa
from project.weak_normal_form import cfg_to_weak_normal_form

__all__ = ["hellings_based_cfpq"]


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return pairs connected by a path in the language of ``cfg``.

    The grammar need not be in weak Chomsky normal form. Both directed
    graphs and directed multigraphs are supported; edge labels are stored
    in the ``label`` attribute. Input objects are not modified.
    """
    normalized = cfg_to_weak_normal_form(cfg)
    automaton = graph_to_nfa(graph, start_nodes, final_nodes)

    triples = {
        (state.value, variable, state.value)
        for state in automaton.states
        for variable in normalized.get_nullable_symbols()
    }
    for source, symbol, target in automaton:
        for production in normalized.productions:
            if (
                len(production.body) == 1
                and isinstance(production.body[0], Terminal)
                and production.body[0].value == symbol.value
            ):
                triples.add((source.value, production.head, target.value))

    while True:
        new_triples = set()
        for source, left, middle in triples:
            for right_source, right, target in triples:
                if middle != right_source:
                    continue
                for production in normalized.productions:
                    if (
                        len(production.body) == 2
                        and production.body[0] == left
                        and production.body[1] == right
                    ):
                        new_triples.add((source, production.head, target))
        new_triples.difference_update(triples)
        if not new_triples:
            break
        triples.update(new_triples)

    starts = {state.value for state in automaton.start_states}
    finals = {state.value for state in automaton.final_states}
    return {
        (source, target)
        for source, variable, target in triples
        if variable == normalized.start_symbol and source in starts and target in finals
    }
