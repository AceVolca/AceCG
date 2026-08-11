"""Real-frame oracle for the force-matching normal equations (F-007).

`local/claude1.md` §3 names "force-matching normal equations: assembly,
conditioning, regularization" as load-bearing, and the suite had no test of it
grounded in real data. This file assembles ``J^T J`` and ``J^T y`` from five
real DOPC frames and the mapped all-atom reference forces those frames already
carry, then drives them through ``FMTrainerAnalytic``.

Two things here can only be learned from real data:

* the model force obeys Newton's third law over a real periodic box, which
  checks every pair/bond/angle projection's sign and minimum image at once;
* the normal equations assembled from a short real slice are **rank
  deficient** — 37 of 192 B-spline columns are never sampled at all. That is
  the "conditioning" half of §3, and it is invisible in a toy where every basis
  function is exercised by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from real_frames import dopc_forcefield, dopc_frame_geometries, dopc_universe

from AceCG.compute.force import force
from AceCG.optimizers.sgd import SGDMaskedOptimizer
from AceCG.trainers.analytic.fm import FMTrainerAnalytic


N_PARAMS = 192
N_FORCE_ROWS = 576 * 3


@pytest.fixture(scope="module")
def forcefield():
    return dopc_forcefield(12)


@pytest.fixture(scope="module")
def geometries(forcefield):
    return dopc_frame_geometries(forcefield)


@pytest.fixture(scope="module")
def reference_forces():
    """The mapped all-atom forces the fixture trajectory already carries."""
    forces = []
    for timestep in dopc_universe().trajectory:
        assert timestep.has_forces
        forces.append(np.asarray(timestep.forces, dtype=np.float64))
    return forces


@pytest.fixture(scope="module")
def normal_equations(forcefield, geometries, reference_forces):
    """Frame-averaged FM statistics, assembled through the production kernel."""
    totals = None
    for geometry, reference in zip(geometries, reference_forces):
        stats = force(
            geometry,
            forcefield,
            return_fm_stats=True,
            reference_force=reference,
        )["fm_stats"]
        if totals is None:
            totals = {
                key: np.array(value, dtype=np.float64)
                if isinstance(value, np.ndarray)
                else float(value)
                for key, value in stats.items()
                if key in {"JtJ", "Jty", "Jtf", "ftf", "fTy", "yty"}
            }
        else:
            for key in totals:
                totals[key] = totals[key] + stats[key]
    n_frames = len(geometries)
    return {key: value / n_frames for key, value in totals.items()}, n_frames


def test_model_force_obeys_newtons_third_law_on_every_real_frame(
    forcefield, geometries
):
    """Σ_i F_i = 0 for the modelled force, on a real periodic configuration.

    Every pair, bond and angle projection contributes an equal and opposite
    pair of rows; a sign error or a dropped minimum image in any one of them
    leaves a residual net force on the box. The reference forces are *not*
    checked this way on purpose — the fixture is a 96-lipid patch cut from a
    1152-lipid bilayer, so the excluded lipids legitimately exert a nonzero net
    force on it.
    """
    for geometry in geometries:
        model = force(geometry, forcefield, return_value=True)["force"]
        assert model.shape == (N_FORCE_ROWS,)
        net = model.reshape(-1, 3).sum(axis=0)
        magnitude = float(np.abs(model).max())
        np.testing.assert_allclose(net, 0.0, atol=1.0e-5 * magnitude)


def test_force_jacobian_is_the_parameter_derivative_of_the_model_force(
    forcefield, geometries
):
    """J[:, k] = ∂F/∂λ_k, checked by central difference on a real frame.

    The Jacobian and the force are stored as float32, so the step is
    deliberately large (1% per parameter): a smaller one buries the signal in
    storage rounding rather than testing anything.
    """
    geometry = geometries[0]
    jacobian = np.asarray(
        force(geometry, forcefield, return_grad=True)["force_grad"], dtype=np.float64
    )
    assert jacobian.shape == (N_FORCE_ROWS, N_PARAMS)

    params = forcefield.param_array().copy()
    scale = np.maximum(1.0, np.abs(params))
    rng = np.random.default_rng(20260810)
    try:
        for _ in range(2):
            direction = rng.standard_normal(params.size)
            direction /= np.linalg.norm(direction)
            displacement = 1.0e-2 * scale * direction

            forcefield.update_params(params + displacement)
            plus = np.asarray(
                force(geometry, forcefield, return_value=True)["force"], dtype=np.float64
            )
            forcefield.update_params(params - displacement)
            minus = np.asarray(
                force(geometry, forcefield, return_value=True)["force"], dtype=np.float64
            )
            forcefield.update_params(params)

            numeric = (plus - minus) / 2.0
            analytic = jacobian @ displacement
            np.testing.assert_allclose(
                analytic, numeric, rtol=1.0e-3, atol=1.0e-4
            )
    finally:
        forcefield.update_params(params)


def test_normal_equations_are_the_documented_contractions(
    forcefield, geometries, reference_forces
):
    """``JtJ``/``Jty``/``Jtf`` and the scalars must be exactly J^T J, J^T y, ...

    Recomputed here from the Jacobian and the two force vectors directly, so a
    transposed einsum or a mixed-up model/reference vector shows up as a number
    rather than as a plausible-looking optimizer trajectory.
    """
    geometry, reference = geometries[0], reference_forces[0]
    result = force(
        geometry,
        forcefield,
        return_value=True,
        return_grad=True,
        return_fm_stats=True,
        reference_force=reference,
    )
    jacobian = np.asarray(result["force_grad"], dtype=np.float64)
    model = np.asarray(result["force"], dtype=np.float64)
    target = reference.reshape(-1)
    stats = result["fm_stats"]

    scale = float(np.abs(jacobian).max()) ** 2 * jacobian.shape[0]
    np.testing.assert_allclose(stats["JtJ"], jacobian.T @ jacobian, atol=1.0e-5 * scale)
    np.testing.assert_allclose(stats["Jty"], jacobian.T @ target, rtol=1.0e-4, atol=1.0e-3)
    np.testing.assert_allclose(stats["Jtf"], jacobian.T @ model, rtol=1.0e-4, atol=1.0e-3)
    assert stats["yty"] == pytest.approx(float(target @ target), rel=1.0e-5)
    assert stats["ftf"] == pytest.approx(float(model @ model), rel=1.0e-5)
    assert stats["fTy"] == pytest.approx(float(model @ target), rel=1.0e-4)
    assert stats["n_force_rows"] == N_FORCE_ROWS
    assert stats["n_atoms_obs"] == N_FORCE_ROWS // 3


def test_normal_equations_from_a_short_real_slice_are_rank_deficient(
    normal_equations,
):
    """Conditioning, measured rather than assumed.

    37 of the 192 B-spline columns are never sampled by these five frames —
    the steep short-r region no real pair reaches, and the far tails. Of the
    155 columns that are sampled, the sub-matrix is still one rank short. This
    is why FM needs masking or regularization and cannot simply be inverted,
    and pinning it means a change that lets an unsampled basis function pick up
    weight (a widened cutoff, a leaked extrapolation) is visible immediately.
    """
    statistics, n_frames = normal_equations
    assert n_frames == 5
    JtJ = statistics["JtJ"]
    assert JtJ.shape == (N_PARAMS, N_PARAMS)

    symmetry_scale = float(np.abs(JtJ).max())
    np.testing.assert_allclose(JtJ, JtJ.T, atol=1.0e-6 * symmetry_scale)

    diagonal = np.diag(JtJ)
    assert np.all(diagonal >= 0.0)
    sampled = np.flatnonzero(diagonal > 0.0)
    assert sampled.size == 155
    assert np.count_nonzero(diagonal <= 0.0) == 37

    block = JtJ[np.ix_(sampled, sampled)]
    assert np.linalg.matrix_rank(block) == 154
    eigenvalues = np.linalg.eigvalsh(block)
    assert eigenvalues.min() > -1.0e-6 * eigenvalues.max()
    assert np.linalg.cond(block) > 1.0e12


def test_fm_trainer_reproduces_the_least_squares_objective_on_real_frames(
    forcefield, normal_equations
):
    """loss = ½‖f − y‖² and grad = J^T f − J^T y, from real statistics.

    ``FMTrainerAnalytic`` never sees the Jacobian, only its contractions, so
    this pins that the contraction bookkeeping and the trainer's algebra agree
    on the same real data.
    """
    statistics, n_frames = normal_equations
    trainer = FMTrainerAnalytic(
        forcefield=forcefield,
        optimizer=SGDMaskedOptimizer(
            forcefield.param_array().copy(),
            np.ones(forcefield.n_params(), dtype=bool),
            lr=0.0,
        ),
    )
    batch = FMTrainerAnalytic.make_batch(
        JtJ=statistics["JtJ"],
        Jty=statistics["Jty"],
        y_sumsq=statistics["yty"],
        Jtf=statistics["Jtf"],
        f_sumsq=statistics["ftf"],
        fty=statistics["fTy"],
        nframe=n_frames,
    )
    out = trainer.step(batch, apply_update=False)

    assert out["name"] == "FM"
    expected_loss = 0.5 * (
        statistics["ftf"] - 2.0 * statistics["fTy"] + statistics["yty"]
    )
    assert out["loss"] == pytest.approx(expected_loss, rel=1.0e-12)
    np.testing.assert_allclose(
        out["grad"], statistics["Jtf"] - statistics["Jty"], rtol=1.0e-12
    )
    # The residual is dominated by the model force, which the untrained
    # B-spline fit puts far from the mapped reference: pin both so a change in
    # either the fit or the reference-force path is visible.
    assert float(statistics["yty"]) == pytest.approx(634665.2375, rel=1.0e-8)
    assert float(statistics["ftf"]) == pytest.approx(8092922.9, rel=1.0e-8)
    assert float(statistics["fTy"]) == pytest.approx(-58996.055469, rel=1.0e-7)
    assert out["loss"] == pytest.approx(4422790.124219, rel=1.0e-8)
