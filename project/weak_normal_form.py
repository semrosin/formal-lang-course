"""Conversion of context-free grammars to weak Chomsky normal form."""

from itertools import count

from pyformlang.cfg import CFG, Production, Terminal, Variable

__all__ = ["cfg_to_weak_normal_form"]


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Return an equivalent grammar with rules ``A -> BC | a | epsilon``.

    Use pyformlang's public unit-production elimination, lift terminals out
    of longer bodies, and split those bodies into binary rules. Epsilon
    productions are retained, including those inherited through unit rules.
    Unlike strict Chomsky normal form, this preserves the empty word.

    Fresh variables cannot collide with the input variables. The input
    grammar is not modified, and its start symbol is preserved.
    """
    without_units = cfg.eliminate_unit_productions()
    variables = set(cfg.variables)
    indices = count()

    def fresh_variable() -> Variable:
        variable = Variable(f"WCNF{next(indices)}")
        while variable in variables:
            variable = Variable(f"WCNF{next(indices)}")
        variables.add(variable)
        return variable

    productions: set[Production] = set()
    terminal_variables: dict[Terminal, Variable] = {}
    for production in without_units.productions:
        if len(production.body) < 2:
            productions.add(production)
            continue

        body = []
        for symbol in production.body:
            if isinstance(symbol, Terminal):
                if symbol not in terminal_variables:
                    terminal_variables[symbol] = fresh_variable()
                    productions.add(Production(terminal_variables[symbol], [symbol]))
                symbol = terminal_variables[symbol]
            body.append(symbol)

        head = production.head
        for symbol in body[:-2]:
            tail = fresh_variable()
            productions.add(Production(head, [symbol, tail]))
            head = tail
        productions.add(Production(head, body[-2:]))

    return CFG(
        variables=variables,
        terminals=cfg.terminals,
        start_symbol=cfg.start_symbol,
        productions=productions,
    )
