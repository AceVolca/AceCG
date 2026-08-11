"""Single-file persistence for a fitted TrajMap force operator.

The force operator persisted here is fit by
:mod:`AceCG.compute.force_mapping`, implementing the linear force-mapping
method of Kraemer, Durumeric, Charron, Chen, Clementi & Noe (2023),
https://doi.org/10.1021/acs.jpclett.3c00444.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..topology.cgmap import CGMapSpec


def write_force_operator(path: Path | str, spec: CGMapSpec, diagnostics: Dict[str, Any]) -> Path:
    """Write one NPZ containing a checked compressed operator and diagnostics."""
    if not spec.has_force_operator:
        raise ValueError("cannot write an authored fixed force map as a fitted operator artifact.")
    target = Path(path)
    if target.suffix.lower() != ".npz":
        target = target.with_suffix(".npz")
    assert spec._force_operator is not None
    matrices, atom_positions, site_positions, coordinate_maps, authored_maps, constraint_pairs, metadata = spec._force_operator
    arrays: Dict[str, Any] = {
        "atom_indices": spec.atom_indices,
        "csr_indptr": spec.csr_indptr,
        "csr_cols": spec.csr_cols,
        "csr_wx": spec.csr_wx,
        "csr_wf": spec.csr_wf,
        "metadata": np.asarray(json.dumps({"operator": dict(metadata), "diagnostics": diagnostics})),
        "n_templates": np.asarray(len(matrices), dtype=np.int64),
    }
    for index, values in enumerate(zip(matrices, atom_positions, site_positions, coordinate_maps, authored_maps, constraint_pairs)):
        prefix = f"template_{index:04d}_"
        for name, value in zip(("matrix", "atoms", "sites", "coordinate", "authored", "pairs"), values):
            arrays[prefix + name] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return target


def read_force_operator(path: Path | str, base_spec: CGMapSpec) -> Tuple[CGMapSpec, Dict[str, Any]]:
    """Attach an NPZ operator to its already compiled compatible mapping spec."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        source = source.with_suffix(".npz")
    with np.load(source, allow_pickle=False) as arrays:
        for name, expected in (("atom_indices", base_spec.atom_indices), ("csr_indptr", base_spec.csr_indptr),
                               ("csr_cols", base_spec.csr_cols), ("csr_wx", base_spec.csr_wx), ("csr_wf", base_spec.csr_wf)):
            if name not in arrays or not np.array_equal(arrays[name], expected):
                raise ValueError(f"force operator is incompatible with CGMapSpec {name}.")
        n_templates = int(arrays["n_templates"])
        blocks = []
        for index in range(n_templates):
            prefix = f"template_{index:04d}_"
            blocks.append(tuple(np.asarray(arrays[prefix + name]) for name in ("matrix", "atoms", "sites", "coordinate", "authored", "pairs")))
        payload = json.loads(str(arrays["metadata"].item()))
    operator = payload.get("operator", {})
    attached = base_spec.with_force_operator(
        [block[0] for block in blocks], [block[1] for block in blocks], [block[2] for block in blocks],
        [block[3] for block in blocks], [block[4] for block in blocks], [block[5] for block in blocks], operator,
    )
    return attached, dict(payload.get("diagnostics", {}))
