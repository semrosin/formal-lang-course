"""Task 2: finite automata construction.

Two conversions are provided:

* :func:`regex_to_dfa` --- builds a *minimal* deterministic finite automaton
  (DFA) that accepts exactly the language described by a regular expression.
* :func:`graph_to_nfa` --- builds a nondeterministic finite automaton (NFA)
  from a labeled directed multigraph, where the graph nodes are the automaton
  states and the edge labels are the input symbols.

"""

from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal DFA that accepts the language of ``regex``."""

    epsilon_nfa = Regex(regex).to_epsilon_nfa()
    return epsilon_nfa.minimize()
