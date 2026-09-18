"""Task 2: finite automata construction.

Two conversions are provided:

* :func:`regex_to_dfa` --- builds a *minimal* deterministic finite automaton
  (DFA) that accepts exactly the language described by a regular expression.
* :func:`graph_to_nfa` --- builds a nondeterministic finite automaton (NFA)
  from a labeled directed multigraph, where the graph nodes are the automaton
  states and the edge labels are the input symbols.

"""

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
    """Build a minimal DFA that accepts the language of ``regex``."""

    epsilon_nfa = Regex(regex).to_epsilon_nfa()
    return epsilon_nfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: set[int] | None = None,
    final_states: set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    """Build an NFA from a labeled directed multigraph."""

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
