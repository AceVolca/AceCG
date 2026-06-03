# 02 Topology Module Developer Reference

*Updated: 2026-04-23.*

The topology layer is built around three core files plus two helper modules:

| File | Responsibility |
|---|---|
| `types.py` | `InteractionKey`, the canonical hash key for all interaction types |
| `topology_array.py` | `TopologyArrays`, a frozen dataclass built once from an MDAnalysis Universe and broadcast to MPI workers |
| `forcefield.py` | `Forcefield`, an `InteractionKey -> List[BasePotential]` dictionary container with masks, bounds, and parameter-vector caches |
| `neighbor.py` | Topology-aware neighbor search, producing `{InteractionKey: (a_idx, b_idx)}` |
| `mscg.py` | MS-CG topology helpers built on the core topology objects |

---

## `InteractionKey`

`InteractionKey` is a `NamedTuple(style, types)` used globally as a dictionary key.

```python
key = InteractionKey.bond("A", "B")
key = InteractionKey.pair("C", "A")     # automatically sorted -> ("A", "C")
key = InteractionKey.angle("X", "Y", "Z")
key = InteractionKey.dihedral("A", "B", "C", "D")
```

### Normalization Rules

All constructors return canonical ordering so that symmetric keys compare equal:

| Style | Rule | Example |
|---|---|---|
| pair | Lexicographic order: `(a, b) if a <= b` | `pair("C", "A")` -> `("A", "C")` |
| bond | Same as pair | Same as above |
| angle | Reverse if `a > c`: `(a, b, c) if a <= c else (c, b, a)` | `angle("Z", "Y", "A")` -> `("A", "Y", "Z")` |
| dihedral | Reverse if `(a, b) > (d, c)` | `dihedral("D", "C", "B", "A")` -> `("A", "B", "C", "D")` |

The center atom of an angle is `b`. The center bond of a dihedral is `b-c`.

### Serialization

```python
key.label()                           # "bond:A:B"
InteractionKey.from_label("bond:A:B") # symmetric deserialization
```

---

## `TopologyArrays`

`TopologyArrays` is a frozen dataclass. It is built once from an MDAnalysis Universe with `collect_topology_arrays()` and then broadcast to all MPI workers. All access is through attributes, not dictionary keys.

### Construction

```python
from AceCG.topology.topology_array import collect_topology_arrays

topo = collect_topology_arrays(
    universe,
    exclude_bonded="111",            # three-character flag: include 1-2, 1-3, 1-4 exclusions
    exclude_option="resid",          # nonbonded exclusion policy: resid / molid / none
    atom_type_name_aliases={1: "CA", 2: "CB"},  # optional LAMMPS type-code -> name map
    vp_names=["VP"],                 # optional virtual-site type names
)
```

### LAMMPS Alias Protocol

`collect_topology_arrays()` can synthesize atom names:

- if `u.atoms.names` exists, use it directly
- if names are absent and `atom_type_name_aliases` is provided, map integer type codes to string names and write them back to the Universe
- if names are absent and no alias is provided, reuse `u.atoms.types` as names and write them back to the Universe

Therefore bonded `InteractionKey` names come from aliases or synthesized names for LAMMPS inputs, and from `u.atoms.types` for topologies that already have names and types.

`atom_type_name_aliases` keys must be integers. Conflicting aliases for the same type id raise an error.

### Field Reference

Atom-level fields:

| Field | Shape / type | Meaning |
|---|---|---|
| `n_atoms` | int | Total atom count, including virtual sites |
| `names` | `(n_atoms,)` str | Per-atom name |
| `types` | `(n_atoms,)` str | Per-atom type name |
| `atom_type_names` | `(n_unique,)` str | Ordered unique type-name list |
| `atom_type_codes` | `(n_atoms,)` int32 | 1-based codes into `atom_type_names` |
| `masses` | `(n_atoms,)` float64 | Atomic masses |
| `charges` | `(n_atoms,)` float64 | Atomic charges |
| `atom_resindex` | `(n_atoms,)` int64 | Residue index for each atom |
| `molnums` | `(n_atoms,)` int64 | Molecule id for each atom; filled with 0 when absent |

Residue-level fields:

| Field | Shape / type | Meaning |
|---|---|---|
| `n_residues` | int | Number of residues |
| `resids` | `(n_residues,)` int64 | Residue ids |

Bonded terms:

| Field | Shape / type | Meaning |
|---|---|---|
| `bonds` | `(n_bond, 2)` int64 | Atom-index pairs |
| `angles` | `(n_angle, 3)` int64 | Atom-index triples |
| `dihedrals` | `(n_dihedral, 4)` int64 | Atom-index quadruples |

Bonded exclusion lists:

| Field | Shape / type | Meaning |
|---|---|---|
| `exclude_12` | `(n_ex, 2)` int64 | Bonded 1-2 pairs |
| `exclude_13` | `(n_ex, 2)` int64 | Angle 1-3 endpoints |
| `exclude_14` | `(n_ex, 2)` int64 | Dihedral 1-4 endpoints |

Pre-encoded exclusion arrays for `neighbor.py`:

| Field | Shape / type | Meaning |
|---|---|---|
| `excluded_nb` | `(n_ex,)` int32 | Encoded excluded pair ids for fast set operations |
| `excluded_nb_mode` | str | Construction-time `exclude_option`: `"resid"`, `"molid"`, or `"none"` |
| `excluded_nb_all` | bool | True means the system is globally excluded, such as a single molecule or residue, so near-neighbor search can be skipped |

`excluded_nb` encodes pair `(a, b)` as `a * n_atoms + b`. `neighbor.py` uses `np.isin()` for vectorized lookup.

Virtual-site classification:

| Field | Shape / type | Meaning |
|---|---|---|
| `real_site_indices` | `(n_real,)` int64 | Non-virtual atom indices |
| `virtual_site_mask` | `(n_atoms,)` bool | True for virtual sites |
| `virtual_site_indices` | `(n_virtual,)` int64 | Virtual-site atom indices |

If `vp_names` is not provided, `real_site_indices = arange(n_atoms)` and `virtual_site_mask` is all false.

Instance-to-type mappings:

| Field | Shape / type | Meaning |
|---|---|---|
| `bond_key_index` | `(n_bond,)` int32 | Each bond instance -> index in `keys_bondtypes` |
| `angle_key_index` | `(n_angle,)` int32 | Each angle instance -> index in `keys_angletypes` |
| `dihedral_key_index` | `(n_dihedral,)` int32 | Each dihedral instance -> index in `keys_dihedraltypes` |
| `keys_bondtypes` | `List[InteractionKey]` | Bond-type key list, in index order |
| `keys_angletypes` | `List[InteractionKey]` | Angle-type key list |
| `keys_dihedraltypes` | `List[InteractionKey]` | Dihedral-type key list |

Conversion dictionaries:

| Field | Type | Meaning |
|---|---|---|
| `atom_type_name_to_code` | `dict[str, int]` | Atom type name -> int code |
| `atom_type_code_to_name` | `dict[int, str]` | Int code -> atom type name |
| `bond_type_id_to_key` | `dict[int, InteractionKey]` | Bond type index -> canonical key |
| `angle_type_id_to_key` | `dict[int, InteractionKey]` | Angle type index -> canonical key |
| `dihedral_type_id_to_key` | `dict[int, InteractionKey]` | Dihedral type index -> canonical key |
| `key_to_bonded_type_id` | `dict[InteractionKey, int]` | Canonical key -> bonded type index |

### Invariants

- The object is frozen after construction.
- All arrays are dense NumPy arrays. Empty topology sections are arrays with shape `(0, width)`, not `None`.
- `bond_key_index[i]` indexes `keys_bondtypes`; both must come from the same Universe.
- `real_site_indices` and `virtual_site_indices` partition `arange(n_atoms)`.
- `exclude_bonded` controls invariant bonded exclusions only: `exclude_12`, `exclude_13`, `exclude_14`.
- Nonbonded exclusion policy lives in `neighbor.py`.
- When `excluded_nb_all=True`, `excluded_nb` is empty as a fast path for single-molecule systems.
- `molnums` is filled with 0 when it is absent from the source topology.

---

## `Forcefield`

`Forcefield` is a `MutableMapping[InteractionKey, List[BasePotential]]` with masks, bounds, and parameter-vector caches.

### Construction

```python
from AceCG.topology.forcefield import Forcefield

ff = Forcefield({key: [pot1, pot2]})  # build from dict
ff2 = Forcefield(ff)                  # copy-construct; shallow potential refs
ff3 = ff.deepcopy()                   # full deep copy
ff4 = copy.deepcopy(ff3)              # also safe
```

### Parameter Vector

The flattened parameter vector is the concatenation of `get_params()` from all potentials in insertion order.

```python
ff.n_params()          # total scalar parameter count
L = ff.param_array()   # (n_params,) float64 copy
ff.update_params(L)    # write back to all potentials and refresh caches
```

Slice helpers:

```python
ff.param_slices()         # [(key, pot_idx, slice), ...]
ff.interaction_offsets()  # [slice, ...]
ff.param_index_map()      # [(key, "k"), (key, "r0"), ...]
```

### Masks

The L2 per-parameter `param_mask` and L1 per-key `key_mask` are synchronized both ways:

```python
ff.param_mask
ff.param_mask = np.array([True, True, False])  # setter derives key_mask

ff.key_mask
ff.key_mask = {bond_key: False}                # setter propagates to param_mask
```

- L1 -> L2: setting `key_mask = {k: False}` clears the entire parameter block for key `k`.
- L2 -> L1: `key_mask[k] = any(param_mask[block])`; a key is active if any parameter in its block is active.

Pattern-based construction:

```python
ff.build_mask(mode="freeze", global_patterns=["*k*"])
ff.build_mask(mode="train", patterns={key: ["r0"]})
```

### Bounds

```python
lb, ub = ff.param_bounds
ff.param_bounds = (lb, ub)
ff.build_bounds(global_bounds={"*k*": (0, None)})
L_safe = ff.apply_bounds(L)
```

Potentials that implement `param_bounds()` automatically contribute bounds during construction.

### VP Masks

VP masks are built once and then frozen because VP status is a predefined topology property:

```python
ff.set_vp_masks(["VP"])
ff.virtual_mask         # all types in key are VP -> True
ff.real_mask            # no type in key is VP -> True
ff.real_virtual_mask    # mixed real / VP key -> True
ff.direct_active_mask   # ~virtual_mask
```

### Key Insertion and Deletion

```python
ff[new_key] = [pot]  # insert: parameter vector grows, caches update locally
del ff[old_key]      # delete: parameter vector shrinks, caches update locally
```

The incremental `_splice_caches` path keeps `param_mask`, `param_bounds`, and VP masks consistent without a full rebuild.

### Iteration

```python
for key in ff:
    ...
for key, pot in ff.iter_potentials():
    ...
```

---

## `neighbor.py`

`neighbor.py` is the topology-layer neighbor-search helper. It consumes raw coordinates and a `TopologyArrays` snapshot and returns atom index pairs or adjacency lists. It does not compute distances, energies, forces, or `FrameGeometry`.

### Public Entry Points

| Function | Responsibility |
|---|---|
| `parse_exclude_option(s)` | Normalize `exclude_option` strings and aliases |
| `compute_pairs_by_type()` | Current engine path: one global neighbor search, binned by canonical `InteractionKey` |
| `compute_neighbor_list()` | Generic per-atom adjacency list; not used by the current compute core |

### Exclusion Protocol

Bonded exclusions always come from `TopologyArrays`:

- `exclude_12`
- `exclude_13`
- `exclude_14`

Additional nonbonded exclusions are selected by `exclude_option`:

| Option | Meaning |
|---|---|
| `resid` | Exclude pairs in the same residue |
| `molid` | Exclude pairs in the same molecule |
| `none` | Do not add extra exclusions |

`parse_exclude_option()` accepts aliases such as `"residue"` -> `"resid"` and `"mol"` -> `"molid"`.

### `compute_pairs_by_type()`

```python
from AceCG.topology.neighbor import compute_pairs_by_type

pair_cache = compute_pairs_by_type(
    positions=positions,
    box=box,
    pair_type_list=pair_keys,
    cutoff=cutoff,
    topology_arrays=topo,
    sel_indices=sel_indices,
    exclude_option="resid",
)
```

Returns:

```python
{InteractionKey: (a_idx, b_idx)}  # a_idx and b_idx are global atom-index arrays
```

### Boundaries and Invariants

- A single global cutoff is used, usually the maximum cutoff across all pair potentials.
- `sel_indices` restricts the search domain but does not renumber atoms.
- Output indices are always global atom indices.
- Per-key cutoff filtering is deferred to `compute_frame_geometry()`.
- This module only performs topology-side routing and should stay numerically simple.

---

## Data-Flow Boundaries

### Ownership

| Data | Owner | Consumers |
|---|---|---|
| `param_mask` (L2) | `Forcefield` | `energy()` / `force()` through `ff.param_mask` |
| `key_mask` (L1) | `Forcefield` | Same as above; engine also passes it to `compute_frame_geometry()` as `interaction_mask` |
| `real_site_indices` | `TopologyArrays` | `compute_frame_geometry()` -> `FrameGeometry` -> `force()` |

### Snapshot Protocol

Before dispatching to the engine, the workflow creates a force-field snapshot carrying the current mask:

```python
ff_snap = Forcefield(self.trainer.forcefield)
```

`Forcefield.param_mask` and `Forcefield.key_mask` are the canonical trainability state. The optimizer receives a mask copy at initialization for execution, but the workflow updates masks on `Forcefield` before taking snapshots.

The engine broadcasts `forcefield_snapshot` and `topology_arrays` through MPI. All per-frame masks are read from those two objects.

### L1 `interaction_mask` Flow in the Engine

The engine reads `forcefield_snapshot.key_mask` once and passes it to `compute_frame_geometry()` as `interaction_mask`, skipping geometry for disabled keys. The same `key_mask` is checked again in `energy()` / `force()` as a safety net.

---

## Allowed Workflow Access Patterns

Do:

- build a Universe for the target topology
- call `collect_topology_arrays()` once whenever topology changes
- pass explicit `atom_type_name_aliases` for LAMMPS inputs
- pass `TopologyArrays` as an immutable snapshot into compute / engine code
- let `neighbor.py` own nonbonded exclusion routing

Do not:

- mutate `TopologyArrays` fields in place
- rebuild topology arrays inside per-frame loops
- invent workflow-local exclusion rules that bypass `exclude_bonded` / `exclude_option`
- treat `frame_id`, trajectory slicing, or MPI rank assignment as topology responsibilities
- reinterpret atom-type aliases after the topology snapshot has been built
