# AceCG/fitters/base.py
from abc import ABC, abstractmethod
from ..potentials.base import BasePotential

class BaseTableFitter(ABC):
    """Interface for turning a LAMMPS table into a potential instance."""
    @abstractmethod
    def fit(self, table_path: str, typ1: str, typ2: str) -> BasePotential:
        ...

    @abstractmethod
    def profile_name(self) -> str:
        ...

class FitterRegistry:
    """Simple registry to construct table fitters by name."""
    def __init__(self):
        self._makers = {}

    def register(self, name: str, maker):
        if name in self._makers:
            raise KeyError(f"Fitter '{name}' already registered")
        self._makers[name] = maker

    def create(self, name: str, **kwargs) -> BaseTableFitter:
        if name not in self._makers:
            raise KeyError(f"Unknown table fitter '{name}'. Available: {list(self._makers)}")
        return self._makers[name](**kwargs)

TABLE_FITTERS = FitterRegistry()