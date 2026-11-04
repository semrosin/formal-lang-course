"""Cases for recursive-automaton conversion and tensor CFPQ."""

from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.finite_automaton import Symbol

from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm


def test_cfg_conversion_preserves_start_symbol_and_productions() -> None:
    cfg = CFG.from_text(
        "Query -> left Query right | $\nUnused -> other",
        start_symbol=Variable("Query"),
    )
    rsm = cfg_to_rsm(cfg)
    box = rsm.get_box(Symbol("Query"))

    assert rsm.initial_label == Symbol("Query")
    assert box.dfa.accepts([])
    assert box.dfa.accepts(["left", "Query", "right"])
    assert not box.dfa.accepts(["left", "right"])


def test_cfg_conversion_preserves_symbols_used_in_regex_syntax() -> None:
    start = Variable("S")
    cfg = CFG(start_symbol=start, productions={Production(start, [Terminal("a|b"), Terminal("*")])})
    box = cfg_to_rsm(cfg).get_box(Symbol("S"))

    assert box.dfa.accepts(["a|b", "*"])
    assert not box.dfa.accepts(["a"])


def test_ebnf_conversion_keeps_regular_expression_structure() -> None:
    rsm = ebnf_to_rsm("S -> (a S b)* | $\nA -> c")

    assert rsm.initial_label == Symbol("S")
    assert rsm.get_box(Symbol("S")).dfa.accepts([])
    assert rsm.get_box(Symbol("S")).dfa.accepts(["a", "S", "b"] * 2)
    assert rsm.get_box(Symbol("A")).dfa.accepts(["c"])
