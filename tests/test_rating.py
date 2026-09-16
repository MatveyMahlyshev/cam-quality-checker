import numpy as np
import pytest

from rating.criteria import CRITERIA, DEFAULT_WEIGHTS, REFERENCE
from rating.weighted import WeightedSumStrategy, normalize_minmax, normalize_reference
from rating.topsis import TopsisStrategy
from rating import build_rating, pairwise_diff


def make_rows(n=4):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {"camera": f"cam{i}"}
        for k in CRITERIA:
            row[k] = float(rng.uniform(0.5, 2.0))
        rows.append(row)
    return rows


def test_normalize_minmax_basic():
    vals = [0, 5, 10]
    out = normalize_minmax(vals)
    assert out == [0.0, 50.0, 100.0]


def test_normalize_minmax_invert():
    vals = [0, 5, 10]
    out = normalize_minmax(vals, invert=True)
    assert out == [100.0, 50.0, 0.0]


def test_normalize_minmax_constant():
    vals = [5, 5, 5]
    out = normalize_minmax(vals)
    assert out == [50.0, 50.0, 50.0]


def test_normalize_reference_above_ref_within_limit():
    out = normalize_reference([1.5], ref=1.0, higher_better=True)
    assert out[0] == 150.0


def test_normalize_reference_clipped_at_150():
    out = normalize_reference([10.0], ref=1.0, higher_better=True)
    assert out[0] == 150.0


def test_normalize_reference_below_ref():
    out = normalize_reference([0.5], ref=1.0, higher_better=True)
    assert out[0] == 50.0


def test_normalize_reference_lower_better():
    out = normalize_reference([0.5], ref=1.0, higher_better=False)
    assert out[0] == 150.0


def test_normalize_reference_clipped_at_150():
    out = normalize_reference([10.0], ref=1.0, higher_better=True)
    assert out[0] == 150.0


def test_weighted_score_count():
    rows = make_rows(4)
    out = WeightedSumStrategy().score(rows)
    assert len(out) == 4
    assert all("score" in r for r in out)


def test_weighted_score_sorted_by_rank():
    rows = make_rows(4)
    rated = build_rating(rows, method="weighted")
    scores = [r["score"] for r in rated]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in rated] == [1, 2, 3, 4]


def test_weighted_empty():
    assert build_rating([], method="weighted") == []


def test_topsis_score_range():
    rows = make_rows(5)
    out = TopsisStrategy().score(rows)
    for r in out:
        assert 0.0 <= r["score"] <= 100.0


def test_topsis_best_has_highest_score():
    rows = make_rows(5)
    for r in rows:
        for k in CRITERIA:
            r[k] = 1.0
    rows[0]["sharpness"] = 10.0
    out = TopsisStrategy().score(rows)
    best = max(out, key=lambda r: r["score"])
    assert best["camera"] == "cam0"


def test_topsis_empty():
    assert build_rating([], method="topsis") == []


def test_build_rating_unknown_method_falls_back_to_weighted():
    rows = make_rows(3)
    out = build_rating(rows, method="unknown")
    assert len(out) == 3


def test_pairwise_diff_count():
    rows = make_rows(4)
    rated = build_rating(rows, method="weighted")
    diffs = pairwise_diff(rated)
    assert len(diffs) == 4 * 3 // 2


def test_pairwise_diff_has_all_criteria():
    rows = make_rows(3)
    rated = build_rating(rows, method="weighted")
    diffs = pairwise_diff(rated)
    for d in diffs:
        for k in CRITERIA:
            assert f"Δ {k}" in d


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_reference_has_all_criteria():
    assert set(REFERENCE.keys()) == set(CRITERIA.keys())