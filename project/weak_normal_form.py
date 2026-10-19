"""Conversion of context-free grammars to weak Chomsky normal form."""

from pyformlang.cfg import CFG, Production

__all__ = ["cfg_to_weak_normal_form"]


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Return an equivalent grammar with rules ``A -> BC | a | epsilon``.

    Restore nullable variables' epsilon rules after pyformlang's conversion
    to Chomsky normal form, then remove useless symbols.
    """
    normal_form_cfg = cfg.to_normal_form()
    nullable = cfg.get_nullable_symbols()

    productions = set(normal_form_cfg.productions)
    productions.update(Production(variable, []) for variable in nullable)

    weak_normal_form_cfg = CFG(start_symbol=cfg.start_symbol, productions=productions)
    return weak_normal_form_cfg.remove_useless_symbols()
