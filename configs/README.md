# Configs

This directory stores reusable AceCG `.acg` input templates and stable config
examples.

## Rules
- Use `.acg` for AceCG workflow inputs.
- Keep reusable templates here.
- Copy a template into `experiments/<level>/<run_id>/inputs/` before launching
  a real run.
- Rewrite all output paths in the run copy so generated files stay inside that
  experiment directory.
- Do not write generated logs, checkpoints, rendered scripts, or raw results
  under `configs/`.

## Current Templates
- `templates/fm_prod.acg`: DOPC force matching template
- `templates/fm_noisy_prod.acg`: DOPC force matching template with
  coordinate-only AA-reference noise enabled
- `templates/rem_prod.acg`: DOPC REM template
- `templates/rem_noisy_prod.acg`: DOPC REM template with coordinate-only
  AA-reference positive-phase noise enabled
- `templates/cdrem_prod.acg`: DOPC CDREM template
- `templates/cdfm_prod.acg`: DOPC CDFM template using paired init config and
  force pools
- `templates/tutorial.acg`: parser reference and commented workflow examples

Some templates still contain paths inherited from the old `workspace/results`
layout. Treat them as starting points and normalize paths in run-local copies.
