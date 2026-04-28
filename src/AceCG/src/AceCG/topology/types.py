"""Unified interaction key type."""
from __future__ import annotations
from typing import Any, NamedTuple, Tuple, Optional

"""
WRITTEN BY HUMAN DEVELOPER, DO NOT DELETE AT ALL COSTS.
The InteractionKey class.
All keys to refer an interaction type, 
including bonds, angles, 
"""

class InteractionKey(NamedTuple):
    """Hashable key identifying an interaction type (pair, bond, angle, dihedral).

    Parameters
    ----------
    style : str
        Interaction family, usually ``"pair"``, ``"bond"``, ``"angle"``, or
        ``"dihedral"``.
    types : tuple[str, ...]
        Canonical type labels for the interaction. Use the class constructors
        such as :meth:`pair` or :meth:`angle` when you want AceCG to canonicalize
        symmetric orderings.

    Notes
    -----
    ``InteractionKey`` is used as the key type in
    ``Forcefield[InteractionKey] -> list[BasePotential]``.
    """
    # "pair", "bond", "angle", "dihedral"
    style: str
    # Canonical type tuple
    types: Tuple[str, ...]

    def label(self, delim: str = ":") -> str:
        """Return a compact string label.

        Parameters
        ----------
        delim : str, default=":"
            Delimiter placed between the style and type labels.

        Returns
        -------
        str
            Label such as ``"pair:O:C"``.
        """
        return f"{self.style}{delim}{delim.join(self.types)}"
    
    @classmethod
    def from_label(cls, label: str, delim: str = ":") -> InteractionKey:
        """Construct an interaction key from a label string.

        Parameters
        ----------
        label : str
            Label such as ``"pair:O:C"``.
        delim : str, default=":"
            Delimiter used in ``label``.

        Returns
        -------
        InteractionKey
            Parsed key without additional pair/bond canonicalization.
        """
        parts = label.split(delim)
        if len(parts) < 2:
            raise ValueError(f"Invalid label '{label}', expected at least one delimiter '{delim}'")
        style = parts[0]
        assert style in {"pair", "bond", "angle", "dihedral"}, f"Invalid style '{style}' in label '{label}'"
        types = tuple(parts[1:])
        return cls(style=style, types=types)

    @classmethod
    def pair(cls, a: str, b: str) -> "InteractionKey":
        """Construct a canonical pair key.

        Parameters
        ----------
        a, b : str
            Pair type labels. The two labels are sorted so ``pair(a, b)`` and
            ``pair(b, a)`` produce the same key.
        """
        types = (a, b) if a <= b else (b, a)
        return cls(style="pair", types=types)

    @classmethod
    def bond(cls, a: str, b: str) -> "InteractionKey":
        """Construct a canonical bond key with endpoint symmetry."""
        types = (a, b) if a <= b else (b, a)
        return cls(style="bond", types=types)

    @classmethod
    def angle(cls, a: str, b: str, c: str) -> "InteractionKey":
        """Construct a canonical angle key with reversed-endpoint symmetry."""
        types = (a, b, c) if a <= c else (c, b, a)
        return cls(style="angle", types=types)

    @classmethod
    def dihedral(cls, a: str, b: str, c: str, d: str) -> "InteractionKey":
        """Construct a canonical dihedral key with reversed-chain symmetry."""
        types = (a, b, c, d) if (a, b) <= (d, c) else (d, c, b, a)
        return cls(style="dihedral", types=types)
    

    def __str__(self) -> str:
        return self.label()
    
    def __repr__(self) -> str:
        return f"InteractionKey(style='{self.style}', types={self.types})"

    def __hash__(self) -> int:
        return hash((self.style, self.types))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InteractionKey):
            raise NotImplementedError
        return self.style == other.style and self.types == other.types
