"""Language preservation and production shapes of weak Chomsky normal form."""

from itertools import product

import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.weak_normal_form import cfg_to_weak_normal_form


@pytest.mark.parametrize(
    "grammar",
    [
        "S -> a",
        "S -> $",
        "S -> A\nA -> B\nB -> a | $",
        "S -> A\nA -> B\nB -> A | b",
        "S -> a S b | $",
        "S -> A B C\nA -> a | $\nB -> b | $\nC -> c | $",
        "S -> S S | a | $",
        "S -> A B\nA -> A\nB -> b",
    ],
)
def test_conversion_preserves_language_and_weak_normal_form(grammar: str) -> None:
    cfg = CFG.from_text(grammar)

    normalized = cfg_to_weak_normal_form(cfg)

    assert normalized.start_symbol == cfg.start_symbol
    for production in normalized.productions:
        body = production.body
        assert (
            not body
            or (len(body) == 1 and isinstance(body[0], Terminal))
            or (len(body) == 2 and all(isinstance(s, Variable) for s in body))
        )
    for length in range(5):
        for word in product("abc", repeat=length):
            assert normalized.contains(word) == cfg.contains(word), word


def test_conversion_preserves_epsilon_from_a_chain_of_nullable_variables() -> None:
    cfg = CFG.from_text("S -> A B\nA -> C\nC -> $\nB -> b | $")

    normalized = cfg_to_weak_normal_form(cfg)

    assert normalized.contains([])
    assert normalized.contains(["b"])
    assert not normalized.contains(["b", "b"])


def test_conversion_removes_unreachable_symbols() -> None:
    cfg = CFG.from_text("S -> a\nUnused -> b | $")

    normalized = cfg_to_weak_normal_form(cfg)

    assert normalized.variables == {Variable("S")}
    assert normalized.terminals == {Terminal("a")}
    assert set(normalized.productions) == {Production(Variable("S"), [Terminal("a")])}


@pytest.mark.parametrize("start_symbol", [None, Variable("S")])
def test_conversion_of_empty_grammar(start_symbol: Variable | None) -> None:
    normalized = cfg_to_weak_normal_form(CFG(start_symbol=start_symbol))

    assert normalized.start_symbol == start_symbol
    assert not normalized.productions
    assert not normalized.contains([])


def test_conversion_does_not_change_the_input_grammar() -> None:
    cfg = CFG.from_text("S -> a S b c | A\nA -> $")
    variables = cfg.variables.copy()
    terminals = cfg.terminals.copy()
    productions = set(cfg.productions)

    cfg_to_weak_normal_form(cfg)

    assert cfg.variables == variables
    assert cfg.terminals == terminals
    assert set(cfg.productions) == productions
