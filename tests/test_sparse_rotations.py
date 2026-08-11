from pathlib import Path

import numpy as np

import crabbymetrics as cm


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sparse_rotation"


def test_varimax_rotation_matches_r_stats_varimax_without_normalization():
    loadings = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.5, 0.4],
        ],
        dtype=float,
    )

    result = cm.varimax_rotation(loadings, normalize=False)

    expected_rotated = np.array(
        [
            [0.80189824483694, 0.192247769629286],
            [0.700935070857621, 0.0932203113158376],
            [0.1087060366428, 0.898989987484519],
            [0.207733494956249, 0.7980268135052],
            [0.503848012063659, 0.395141975420859],
        ]
    )
    expected_rotation = np.array(
        [
            [0.999953161463837, -0.00967857832935167],
            [0.00967857832935167, 0.999953161463837],
        ]
    )
    expected_objective = 0.846932426665476

    np.testing.assert_allclose(result["rotated"], expected_rotated, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        result["rotation"], expected_rotation, atol=1e-12, rtol=1e-12
    )
    assert abs(result["objective"] - expected_objective) < 1e-12
    assert result["converged"]


def test_varimax_rotation_matches_r_stats_varimax_with_kaiser_normalization():
    loadings = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.5, 0.4],
        ],
        dtype=float,
    )

    result = cm.varimax_rotation(loadings, normalize=True)

    expected_rotated = np.array(
        [
            [0.805570332575399, 0.176228372501031],
            [0.702653669476166, 0.0792327001412854],
            [0.126600626057179, 0.896645014195657],
            [0.223596298416925, 0.793728351096425],
            [0.511622820126418, 0.385022194069239],
        ]
    )
    expected_rotation = np.array(
        [
            [0.99956167729489, -0.0296049536974336],
            [0.0296049536974336, 0.99956167729489],
        ]
    )

    np.testing.assert_allclose(result["rotated"], expected_rotated, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        result["rotation"], expected_rotation, atol=1e-12, rtol=1e-12
    )


def test_l1_sparse_rotation_collates_freyaldenhoven_reference_candidates():
    loadings = np.loadtxt(FIXTURE_DIR / "Lambda_ex1.csv", delimiter=",")
    candidate_directions = np.loadtxt(FIXTURE_DIR / "R_ex1.csv", delimiter=",")
    expected_rotated = np.loadtxt(
        FIXTURE_DIR / "ex1_rot_mat_matlab.csv", delimiter=","
    )

    result = cm.l1_sparse_rotation(
        loadings,
        initial_directions=candidate_directions,
        tol=1e-8,
    )

    expected_first_rows = np.array(
        [
            [2.7233906793, 1.4719001932, 0.2966748757, 0.2840247081],
            [0.0547494334, -0.0573416877, 0.7983949701, 1.5627511998],
            [-0.0484066891, 1.8559360622, -0.0334844626, 0.0763419567],
            [-0.0176473148, -0.1163063338, -0.8141478036, 0.853255017],
            [-0.1049942917, -0.1804150346, -0.0582278577, 1.3284931654],
            [-0.1280912531, -0.2974486337, -0.0762094541, 1.3956877137],
        ]
    )
    expected_l1 = np.array(
        [142.354393050604, 156.340795186554, 191.090122249862, 242.290538674616]
    )

    np.testing.assert_allclose(
        result["rotated"][:6, :], expected_first_rows, atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.sum(np.abs(result["rotated"]), axis=0), expected_l1, atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        result["rotated"], expected_rotated, atol=1e-4, rtol=1e-8
    )
    assert result["rotation"].shape == (4, 4)
    assert result["candidate_directions"].shape == candidate_directions.shape


def test_sparse_rotation_diagnostics_match_reference_formulas():
    loadings = np.array(
        [
            [0.01, 0.8],
            [0.02, 0.7],
            [0.03, 0.1],
            [0.7, 0.02],
            [0.8, 0.01],
            [0.9, 0.03],
        ],
        dtype=float,
    )

    counts = cm.count_small_loadings(loadings, threshold=0.05)
    diagnostic = cm.local_factor_diagnostic(loadings, threshold=0.05)
    ipr = cm.inverse_participation_ratio(loadings)
    cumulative = cm.cumulative_participation(loadings)

    np.testing.assert_array_equal(counts, np.array([3.0, 3.0]))
    np.testing.assert_array_equal(diagnostic["n_small"], np.array([3.0, 3.0]))
    assert diagnostic["gamma_n"] == 1.0
    assert diagnostic["has_local_factors"]
    np.testing.assert_allclose(ipr, np.array([1.30580098, 0.64980098]), atol=1e-12)
    assert abs(cumulative - 1.95811796) < 1e-12
