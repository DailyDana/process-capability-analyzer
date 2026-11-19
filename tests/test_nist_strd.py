"""Numerical accuracy against the NIST Statistical Reference Datasets.

NIST certifies the mean, the standard deviation and the lag-1 autocorrelation
of nine univariate datasets to 15 significant digits. These test the
arithmetic, not the control charts: NumAcc2-4 (large offsets, tiny spread)
are built to break a variance computed as ``mean(x^2) - mean(x)^2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.statistics.autocorrelation import lag1_autocorrelation
from app.statistics.descriptive import describe
from app.statistics.variation import grand_mean, sigma_overall
from tests.fixtures import nist

# Double precision carries ~15.9 digits; the NumAcc sets lose some to the
# offset (NumAcc4: 1e7 against a spread of 0.1), so they get a looser bound.
REL = {"NumAcc2": 1e-13, "NumAcc3": 1e-9, "NumAcc4": 1e-8}
DEFAULT_REL = 1e-13


@pytest.fixture(scope="module", params=nist.NAMES)
def ds(request: pytest.FixtureRequest) -> nist.NistDataset:
    return nist.load(request.param)


def test_mean(ds: nist.NistDataset) -> None:
    assert grand_mean(ds.values) == pytest.approx(ds.mean, rel=1e-14)
    assert describe(ds.values).mean == pytest.approx(ds.mean, rel=1e-14)


def test_standard_deviation(ds: nist.NistDataset) -> None:
    rel = REL.get(ds.name, DEFAULT_REL)
    assert sigma_overall(ds.values) == pytest.approx(ds.sd, rel=rel)
    assert describe(ds.values).std_dev == pytest.approx(ds.sd, rel=rel)


def test_lag1_autocorrelation(ds: nist.NistDataset) -> None:
    rel = REL.get(ds.name, DEFAULT_REL)
    assert lag1_autocorrelation(ds.values).r1 == pytest.approx(ds.r1, rel=max(rel, 1e-12))


def test_all_nine_datasets_are_present() -> None:
    assert len(nist.NAMES) == 9
    assert sum(len(nist.load(n).values) for n in nist.NAMES) == 100 + 50 + 200 + 218 + 5000 + 3 + 3 * 1001


def test_a_naive_variance_fails_on_numacc4() -> None:
    """Shows the test has teeth: the textbook shortcut is wrong by orders of magnitude here."""
    x = np.asarray(nist.load("NumAcc4").values)
    naive = np.sqrt(max(0.0, (np.mean(x * x) - np.mean(x) ** 2) * x.size / (x.size - 1)))
    assert abs(naive - 0.1) / 0.1 > 1e-3
