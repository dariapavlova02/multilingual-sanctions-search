"""Compatibility names for configuration fields renamed during refactoring."""

from functools import wraps

FLAG_ALIASES = {
    "fsm_tuned_roles": "enable_fsm_tuned_roles",
    "enhanced_diminutives": "enable_enhanced_diminutives",
    "enhanced_gender_rules": "enable_enhanced_gender_rules",
    "ascii_fastpath": "enable_ascii_fastpath",
}


def canonical_flag_names(values):
    """Resolve aliases and reject contradictory values instead of ignoring them."""
    resolved = dict(values)
    for alias, canonical in FLAG_ALIASES.items():
        if alias in resolved:
            value = resolved.pop(alias)
            if canonical in resolved and resolved[canonical] != value:
                raise ValueError(f"Conflicting values for {alias} and {canonical}")
            resolved[canonical] = value
    return resolved


def accept_flag_aliases(cls):
    """Keep one stored field while accepting historical constructor/access names."""
    original_init = cls.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **canonical_flag_names(kwargs))

    cls.__init__ = init
    for alias, canonical in FLAG_ALIASES.items():
        if canonical in cls.__dataclass_fields__:
            setattr(
                cls,
                alias,
                property(
                    lambda self, field=canonical: getattr(self, field),
                    lambda self, value, field=canonical: setattr(self, field, value),
                ),
            )
    return cls
