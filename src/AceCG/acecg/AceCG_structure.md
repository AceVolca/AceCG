# 📦 AceCG Package Structure

```
AceCG/                         # Main package root
├── __init__.py               # Public API shortcuts for core components
├── optimizers/               # Optimizers for REM parameter updates
│   ├── base.py               # Abstract BaseOptimizer interface
│   └── newton_raphson.py     # Newton-Raphson optimizer using masked Hessian
├── potentials/               # Coarse-grained potential function modules
│   ├── base.py               # Abstract BasePotential interface
│   └── gaussian.py           # Gaussian pair potential implementation
├── trainers/                 # REM training logic (analytic & NN)
│   ├── base.py               # BaseREMTrainer interface for strategy pattern
│   ├── analytic.py           # REMTrainer for analytical (non-NN) potentials
│   └── utils.py              # Helper: prepare_REM_data(), trajectory wrappers
├── utils/                    # Utility modules for REM and FF I/O
│   ├── compute.py            # dU/dL, Hessian, Fisher matrix construction
│   ├── ffio.py               # Read/write LAMMPS pair_coeff, param flattening
│   └── neighbor.py           # Neighbor list + pairwise distances per frame
```