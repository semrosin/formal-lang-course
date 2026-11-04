"""Recursive automata and tensor-based context-free path querying."""

from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Epsilon, EpsilonNFA, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Build one finite automaton for the productions of each variable."""
    boxes = set()
    for variable in cfg.variables | {cfg.start_symbol}:
        automaton = EpsilonNFA()
        automaton.add_start_state(0)
        next_state = 1
        for production in cfg.productions:
            if production.head != variable:
                continue
            state = 0
            for symbol in production.body:
                automaton.add_transition(state, Symbol(symbol.value), next_state)
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
