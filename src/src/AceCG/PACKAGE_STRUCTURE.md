# AceCG Package Structure

This file gives a source-level map of the `AceCG` package. The tree below uses tab characters for indentation.

```text
AceCG/	# Main package root
├── __init__.py	# Public API shortcuts for core components
├── PACKAGE_STRUCTURE.md	# Source package structure overview
├── analysis/	# Analysis utilities for distributions and observables
│	├── __init__.py	# Analysis package exports
│	└── rdf.py	# Pair, bond, angle, dihedral, and interaction distribution tools
├── compute/	# Frame-level compute engine and reducers
│	├── __init__.py	# Compute package exports
│	├── energy.py	# Energy, energy-gradient, Hessian, and gradient-outer computations
│	├── force.py	# Force, force-gradient, and force-matching statistics
│	├── frame_geometry.py	# Per-frame geometry extraction from topology and coordinates
│	├── mpi_engine.py	# MPI-aware compute engine and registered observable execution
│	├── reducers.py	# One-pass reducer helpers for FM, REM, RDF, cache, and CDFM batches
│	├── registry.py	# Default compute-engine builder and observable registrations
│	└── vp_prepare.py	# Virtual-particle preparation helpers for compute workflows
├── configs/	# ACG and VP configuration parsing
│	├── __init__.py	# Config package exports
│	├── models.py	# Frozen dataclass models for parsed AceCG configs
│	├── parser.py	# `.acg` parser and validation logic
│	├── utils.py	# Config parsing utilities
│	├── vp_config.py	# Virtual-particle topology config parser/model
│	└── vp_growth_config.py	# VP-growth workflow config parser/model
├── fitters/	# Table-to-potential fitting utilities
│	├── __init__.py	# Fitter registry setup
│	├── base.py	# Base table-fitter interface and registry
│	├── fit_bspline.py	# B-spline force-basis fitter for LAMMPS tables
│	├── fit_harmonic.py	# Harmonic fitter for bond/angle tables
│	├── fit_multi_gaussian.py	# Multi-Gaussian fitter for pair tables
│	└── utils.py	# Shared fitter numerical helpers
├── io/	# File I/O, logging, table export, coordinates, and trajectories
│	├── __init__.py	# Public I/O exports
│	├── coordinates.py	# AA-to-CG coordinate builder from mapping YAML
│	├── coordinates_writers.py	# GRO, PDB, and LAMMPS data writers
│	├── forcefield.py	# LAMMPS forcefield read/write helpers
│	├── logger.py	# Small timestamped screen logger
│	├── tables.py	# LAMMPS table parsing, writing, conversion, and comparison
│	├── trajectory.py	# LAMMPS trajectory loading and splitting helpers
│	└── vp_ffbuilder.py	# VP forcefield-building helpers
├── optimizers/	# Masked optimizers over the global parameter vector
│	├── __init__.py	# Optimizer exports
│	├── adam.py	# Masked Adam optimizer
│	├── adamW.py	# Masked AdamW optimizer
│	├── base.py	# Base optimizer interface
│	├── newton_raphson.py	# Masked Newton-Raphson optimizer
│	├── rmsprop.py	# Masked RMSprop optimizer
│	└── multithreaded/	# Optional faster optimizer variants
│		├── __init__.py	# Multithreaded optimizer exports
│		└── adam.py	# Numba-parallel masked Adam optimizer
├── potentials/	# Analytic potential models and parameter derivatives
│	├── __init__.py	# Potential exports and LAMMPS-style registry
│	├── base.py	# BasePotential interface and potential iterator helper
│	├── bspline.py	# Force-basis B-spline potential
│	├── gaussian.py	# Single normalized Gaussian potential
│	├── harmonic.py	# Harmonic bond/angle potential
│	├── lennardjones.py	# Lennard-Jones 12-6 potential
│	├── lennardjones96.py	# Lennard-Jones 9-6 potential
│	├── lennardjones_soft.py	# Soft-core Lennard-Jones potential
│	├── multi_gaussian.py	# Sum of normalized Gaussian components
│	├── soft.py	# Cosine-soft pair potential
│	├── srlrgaussian.py	# Short-range/long-range Gaussian potential
│	└── unnormalized_multi_gaussian.py	# LAMMPS double/gauss-compatible multi-Gaussian potential
├── samplers/	# Simulation input staging and conditioned sampling
│	├── __init__.py	# Sampler package exports
│	├── _lammps_script.py	# Lightweight LAMMPS input-script parser
│	├── _script_inspector.py	# Backend-neutral script inspection protocol
│	├── base.py	# Base sampler and epoch/run state records
│	└── conditioned.py	# Conditioned sampler for z|x style tasks
├── schedulers/	# CPU lease management and task launching
│	├── __init__.py	# Scheduler public API
│	├── mpi_backend.py	# MPI backend abstraction for Intel MPI, OpenMPI, MPICH, and local mpirun
│	├── profiler.py	# Preflight MPI benchmark helper
│	├── resource_pool.py	# Host discovery, CPU leases, placement, and resource pool
│	├── task_runner.py	# Worker-side task execution entry point
│	└── task_scheduler.py	# Controller-side streaming task scheduler
├── solvers/	# Closed-form solvers for statistics batches
│	├── __init__.py	# Solver exports
│	├── base.py	# Base solver interface
│	└── fm_matrix.py	# OLS, ridge, and Bayesian FM matrix solver
├── topology/	# Topology keys, forcefield container, neighbor lists, and VP growth
│	├── __init__.py	# Topology public API
│	├── forcefield.py	# Canonical InteractionKey-to-potential-list forcefield container
│	├── mscg.py	# MSCG topology parsing and replicated topology helpers
│	├── neighbor.py	# Pair and neighbor-list construction helpers
│	├── topology_array.py	# Immutable topology arrays for MPI workers
│	├── types.py	# InteractionKey type and canonical constructors
│	└── vpgrower.py	# Virtual-particle topology template and per-frame growth
├── trainers/	# Optimization trainers that consume compute statistics
│	├── __init__.py	# Trainer exports
│	├── base.py	# BaseTrainer interface and shared parameter helpers
│	├── analytic/	# NumPy/statistics-based trainers
│	│	├── __init__.py	# Analytic trainer exports
│	│	├── cdfm.py	# Conditional force-matching trainer
│	│	├── cdrem.py	# Conditional/latent relative-entropy trainer
│	│	├── fm.py	# Force-matching gradient trainer
│	│	├── mse.py	# PMF-matching MSE trainer
│	│	├── multi.py	# Meta-trainer combining multiple trainers
│	│	└── rem.py	# Relative entropy minimization trainer
│	└── autodiff/	# Placeholder for autodiff trainer implementations
│		└── __init__.py	# Autodiff package marker
└── workflows/	# End-to-end workflow drivers
	├── __init__.py	# Workflow package exports
	├── base.py	# Shared config, topology, forcefield, optimizer, and resource builders
	├── cdfm.py	# CDFM production workflow
	├── cdrem.py	# CDREM production workflow
	├── fm.py	# Force-matching workflow
	├── rem.py	# REM workflow
	├── sampling.py	# Shared sampling-workflow base class
	└── vp_growth.py	# VP growth workflow and CLI entry points
```

## Layer Guide

- `configs`, `io`, and `topology` prepare structured inputs: parsed config, forcefield parameters, topology arrays, and simulation files.
- `potentials` defines the analytic parameterized functions used by `forcefield`.
- `compute` turns trajectory frames into reduced statistics for trainers and solvers.
- `trainers`, `optimizers`, and `solvers` update forcefield parameters.
- `samplers` and `schedulers` stage and run simulation tasks.
- `workflows` connect all layers into runnable FM, REM, CDREM, CDFM, and VP-growth pipelines.
