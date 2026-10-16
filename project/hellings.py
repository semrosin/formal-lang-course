"""Hellings' algorithm for context-free path queries."""

from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.cfg import CFG, Variable

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

    Start/final handling is inherited from :func:`graph_to_nfa`: ``None``
    or an empty set selects all graph vertices; explicitly supplied absent
    vertices are isolated states. Nullable rules seed paths of length zero.
    Restrictions apply to the returned endpoints, allowing any intermediate
    vertex on a matching path.

    Each discovered ``(variable, source, target)`` triple is queued once.
    Grammar and path indexes limit joins to compatible rules and endpoints.
    """
    normalized = cfg_to_weak_normal_form(cfg)
    automaton = graph_to_nfa(graph, start_nodes, final_nodes)

    terminal_rules: dict[object, set[Variable]] = defaultdict(set)
    nullable: set[Variable] = set()
    rules_by_left: dict[Variable, list[tuple[Variable, Variable]]] = defaultdict(list)
    rules_by_right: dict[Variable, list[tuple[Variable, Variable]]] = defaultdict(list)
    for production in normalized.productions:
        if not production.body:
            nullable.add(production.head)
        elif len(production.body) == 1:
            terminal_rules[production.body[0].value].add(production.head)
        else:
            left, right = production.body
            rules_by_left[left].append((production.head, right))
            rules_by_right[right].append((production.head, left))

    successors: dict[tuple[Variable, int], set[int]] = defaultdict(set)
    predecessors: dict[tuple[Variable, int], set[int]] = defaultdict(set)
    pending: deque[tuple[Variable, int, int]] = deque()

    def add_path(variable: Variable, source: int, target: int) -> None:
        targets = successors[variable, source]
        if target not in targets:
            targets.add(target)
            predecessors[variable, target].add(source)
            pending.append((variable, source, target))

    for variable in nullable:
        for state in automaton.states:
            add_path(variable, state.value, state.value)
    for source, symbol, target in automaton:
        for variable in terminal_rules.get(symbol.value, ()):
            add_path(variable, source.value, target.value)

    while pending:
        variable, source, target = pending.popleft()
        # Recursive rules can enlarge a path set during a join: iterate a snapshot.
        for head, left in rules_by_right.get(variable, ()):
            for previous in tuple(predecessors.get((left, source), ())):
                add_path(head, previous, target)
        for head, right in rules_by_left.get(variable, ()):
            for following in tuple(successors.get((right, target), ())):
                add_path(head, source, following)

    finals = {state.value for state in automaton.final_states}
    return {
        (start.value, target)
        for start in automaton.start_states
        for target in successors.get((normalized.start_symbol, start.value), ())
        if target in finals
    }
