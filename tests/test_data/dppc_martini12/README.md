# DPPC Martini-12 mapping fixture

The AA-to-CG mapping specification and Martini-2.0 topology used to test
`topology/cgmap.py`'s mapping-YAML compiler and `topology/cgmap_builder.py`'s
residue-topology builder against a real 12-site Martini mapping, without
depending on the group's private trajectory archive.

## Provenance

| Field | Value |
|---|---|
| `map.yaml` | Group trajectory-archive mapping spec for DPPC AA -> Martini-12 CG, staged root for `dppc_aa_Pak2019_4608` (see `data/catalog.yaml`) |
| `martini_v2.0_DPPC_01-alt-opt.itp` | Martini v2.0 DPPC topology, alternate-optimized variant, previously tracked under `human_only/conversations/cgmap_chatlogs/` |
| Extraction date | 2026-08-11 |
| Rights | Both files are the author's own / lab-internal; cleared for redistribution 2026-08-11 |

Both files are small (under 4 KB combined) and copied verbatim, no
recomputation. No hashing.

## Rules for tests that use this

* Load through the production path (`load_mapping_yaml`, `parse_gromacs_itp`),
  not by hand-parsing.
* These replace the `trajmap_archive_paths.py` private-archive locator that
  used to gate `tests/test_trajmap_{builder,kernel,spec}.py`'s `map.yaml` /
  Martini-itp coverage behind a `skipif` — that locator hardcoded a
  lab-internal NFS path and a labmate's name and was never meant to be
  published; these tests now run unconditionally instead of silently
  skipping on any machine without that private mount.
