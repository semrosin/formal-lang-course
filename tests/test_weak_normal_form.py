"""Language preservation and production shapes of weak Chomsky normal form."""

import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.weak_normal_form import cfg_to_weak_normal_form


@pytest.mark.parametrize(
    "grammar, accepted_words, rejected_words",
    [
        (
            "S -> A B\nA -> C\nC -> $ | a\nB -> b | $",
            [(), ("a",), ("b",), ("a", "b")],
            [("b", "a"), ("a", "a"), ("b", "b")],
        ),
        (
            "S -> a S b c | $",
            [(), ("a", "b", "c"), ("a", "a", "b", "c", "b", "c")],
            [("a", "b"), ("a", "b", "c", "c"), ("a", "b", "c", "a", "b", "c")],
        ),
    ],
)
def test_conversion_preserves_language_and_weak_normal_form(
    grammar: str, accepted_words, rejected_words
) -> None:
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
    for word in accepted_words:
        assert normalized.contains(word), word
    for word in rejected_words:
        assert not normalized.contains(word), word


def test_conversion_removes_unreachable_symbols() -> None:
    cfg = CFG.from_text("S -> a\nUnused -> b | $")

    normalized = cfg_to_weak_normal_form(cfg)

    assert normalized.variables == {Variable("S")}
    assert normalized.terminals == {Terminal("a")}
    assert set(normalized.productions) == {Production(Variable("S"), [Terminal("a")])}


def test_conversion_of_empty_grammar() -> None:
    start_symbol = Variable("S")
    normalized = cfg_to_weak_normal_form(CFG(start_symbol=start_symbol))

    assert normalized.start_symbol == start_symbol
    assert not normalized.productions
    assert not normalized.contains([])
