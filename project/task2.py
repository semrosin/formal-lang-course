"""Task 2: finite automata construction.

Two conversions are provided:

* :func:`regex_to_dfa` --- builds a *minimal* deterministic finite automaton
  (DFA) that accepts exactly the language described by a regular expression.
* :func:`graph_to_nfa` --- builds a nondeterministic finite automaton (NFA)
  from a labeled directed multigraph, where the graph nodes are the automaton
  states and the edge labels are the input symbols.

"""

from typing import Set

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
)
from pyformlang.regular_expression import Regex

__all__ = ["graph_to_nfa", "regex_to_dfa"]

#: Name of the edge attribute that stores the label (input symbol) of an edge.
LABEL_ATTRIBUTE = "label"


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal DFA that accepts the language of ``regex``.

    The syntax is the one of :class:`pyformlang.regular_expression.Regex`:
    ``|`` is union, ``*`` is the Kleene star, juxtaposition is concatenation,
    parentheses group sub-expressions, and whitespace between the tokens is
    ignored.
    Symbols are arbitrary strings, so multi-character symbols such as
    ``"a1"`` are allowed.

    Args:
        regex: A regular expression, for example ``"(a | b)*c"``.

    Returns:
        :class:`~pyformlang.finite_automaton.DeterministicFiniteAutomaton`
        that accepts exactly the words described by ``regex``. The automaton
        is deterministic and minimal; in particular it accepts the empty word
        exactly when ``regex`` does.

    Raises:
        MisformedRegexError: If ``regex`` cannot be parsed. The exception is defined in
            :mod:`pyformlang.regular_expression`.

    Examples:
        >>> dfa = regex_to_dfa("a (b | c)*")
        >>> dfa.accepts(["a", "b", "c"])
        True
        >>> dfa.accepts(["a", "d"])
        False
        >>> dfa.is_deterministic()
        True
    """

    epsilon_nfa = Regex(regex).to_epsilon_nfa()
    return epsilon_nfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: Set[int],
    final_states: Set[int],
) -> NondeterministicFiniteAutomaton:
    """Build an NFA from a labeled directed multigraph.

    The graph is used as is: every node becomes an automaton state, and every
    edge ``u --label--> v`` becomes a transition that reads the symbol
    ``label`` and goes from ``u`` to ``v``.

    Args:
        graph: A :class:`networkx.MultiDiGraph` whose edges carry the
            :data:`LABEL_ATTRIBUTE` (``"label"``) attribute with the input
            symbol. Nodes can be of any hashable type, but the types used in
            the tests are integers and strings.
        start_states: Nodes that become the start states of the automaton. An
            empty set means "all nodes of ``graph`` are start states".
        final_states: Nodes that become the final states of the automaton. An
            empty set means "all nodes of ``graph`` are final states".

    Returns:
        :class:`~pyformlang.finite_automaton.NondeterministicFiniteAutomaton`
        whose states are the nodes of ``graph`` and whose transitions are the
        edges of ``graph``.

    Examples:
        >>> import networkx as nx
        >>> graph = nx.MultiDiGraph()
        >>> _ = graph.add_edges_from([(0, 1), (1, 2)], label="a")
        >>> nfa = graph_to_nfa(graph, {0}, {2})
        >>> nfa.accepts(["a", "a"])
        True
        >>> nfa.accepts(["a"])
        False
    """

    nodes = set(graph.nodes)

    starts = set(start_states) if start_states else nodes
    finals = set(final_states) if final_states else nodes

    nfa = NondeterministicFiniteAutomaton(
        states=nodes,
        start_state=starts,
        final_states=finals,
    )

    nfa.add_transitions(
        (source, Symbol(label), target)
        for source, target, label in graph.edges(data=LABEL_ATTRIBUTE)
    )

    return nfa
