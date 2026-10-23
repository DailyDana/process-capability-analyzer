"""Non-normal capability (app.capability.nonnormal).

Evidence:
1. Published: the iSixSigma cycle-time study (n = 207, USL 30, lower bound 0)
   -- Weibull 233244.81 PPM, bounded normal 249535.66 PPM and Box-Cox
   (lambda 0.5) 227113.29 PPM, all reproduced to the printed decimal.
2. Exact maximum likelihood: closed forms (lognormal, exponential) and the
   score equations (Weibull, gamma).
3. Consistency: the percentile method on a normal fit equals classic Pp/Ppk;
   percentiles of exactly specified distributions are analytic.
4. Monte Carlo: Ppk estimates converge on the true Ppk.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from app.capability.nonnormal import (
    P_HIGH,
    P_LOW,
    Distribution,
    NonNormalSpec,
    anderson_darling,
    box_cox,
    boxcox_transform,
    fit_all,
    fit_distribution,
    normal_theory_ppk,
    percentile_indices,
)

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "nonnormal_cycle_time.csv"


@pytest.fixture(scope="module")
def cycle():
    with open(EXAMPLE, newline="", encoding="utf-8") as fh:
        return np.array([float(r["cycle_time_days"]) for r in csv.DictReader(fh)])


UPPER_ONLY = NonNormalSpec(usl=30.0, lsl=0.0, lsl_is_boundary=True)


class TestPublishedCycleTime:
    def test_data_matches_published_summary(self, cycle):
        # Source Table 5: N 207, mean 21.787, StDev 12.135
        assert cycle.size == 207
        assert cycle.mean() == pytest.approx(21.787, abs=5e-4)
        assert cycle.std(ddof=1) == pytest.approx(12.135, abs=5e-4)

    def test_weibull_ppm(self, cycle):
        fit = fit_distribution(cycle, Distribution.WEIBULL)
        assert percentile_indices(fit, UPPER_ONLY).ppm_total == pytest.approx(233244.81, abs=0.005)

    def test_bounded_normal_ppm_with_c4(self, cycle):
        # The published figure uses s / c4 (Minitab's default unbiasing).
        fit = fit_distribution(cycle, Distribution.NORMAL, unbias_c4=True)
        assert percentile_indices(fit, UPPER_ONLY).ppm_total == pytest.approx(249535.66, abs=0.005)
        plain = fit_distribution(cycle, Distribution.NORMAL)
        assert percentile_indices(plain, UPPER_ONLY).ppm_total == pytest.approx(249275.15, abs=0.005)

    def test_box_cox_ppm(self, cycle):
        r = box_cox(cycle, UPPER_ONLY, unbias_c4=True)
        assert r.lambda_mle == pytest.approx(0.5064, abs=5e-5)
        assert r.lambda_used == 0.5  # the published "lambda of 0.5"
        assert r.ppm_total == pytest.approx(227113.29, abs=0.005)

    def test_weibull_is_best_fit(self, cycle):
        # The article's conclusion: Weibull describes the data; normal does not.
        fits, _ = fit_all(cycle)
        assert fits[0].distribution is Distribution.WEIBULL
        ranks = [f.distribution for f in fits]
        assert ranks.index(Distribution.WEIBULL) < ranks.index(Distribution.NORMAL)


class TestExactMle:
    rng = np.random.default_rng(2026)

    def test_lognormal_closed_form(self):
        x = self.rng.lognormal(1.0, 0.4, 300)
        f = fit_distribution(x, Distribution.LOGNORMAL)
        assert f.params["mu (log)"] == pytest.approx(np.log(x).mean(), abs=1e-12)
        assert f.params["sigma (log)"] == pytest.approx(np.log(x).std(ddof=0), abs=1e-12)

    def test_exponential_closed_form(self):
        x = self.rng.exponential(3.0, 300)
        assert fit_distribution(x, Distribution.EXPONENTIAL).params["scale"] == pytest.approx(x.mean(), abs=1e-12)

    def test_weibull_score_equations(self):
        x = self.rng.weibull(2.5, 400) * 7.0
        p = fit_distribution(x, Distribution.WEIBULL).params
        k, lam = p["shape"], p["scale"]
        lx = np.log(x)
        assert np.sum(x**k * lx) / np.sum(x**k) - 1 / k - lx.mean() == pytest.approx(0.0, abs=1e-10)
        assert lam**k == pytest.approx(np.mean(x**k), rel=1e-10)

    def test_weibull_beats_generic_optimizer_on_likelihood(self, cycle):
        f = fit_distribution(cycle, Distribution.WEIBULL)
        c, _, s = stats.weibull_min.fit(cycle, floc=0)
        assert f.log_likelihood >= float(np.sum(stats.weibull_min.logpdf(cycle, c, 0, s))) - 1e-9

    def test_gamma_score_equation(self):
        x = self.rng.gamma(3.0, 2.0, 400)
        p = fit_distribution(x, Distribution.GAMMA).params
        k = p["shape"]
        assert np.log(k) - special.digamma(k) == pytest.approx(np.log(x.mean()) - np.log(x).mean(), abs=1e-10)
        assert k * p["scale"] == pytest.approx(x.mean(), rel=1e-12)

    def test_positive_only_support(self):
        x = np.array([-1.0, 1.0, 2.0, 3.0])
        for d in (Distribution.LOGNORMAL, Distribution.WEIBULL, Distribution.GAMMA, Distribution.EXPONENTIAL):
            with pytest.raises(ValueError, match="positive"):
                fit_distribution(x, d)
        fits, skipped = fit_all(x)
        assert [f.distribution for f in fits] == [Distribution.NORMAL] and len(skipped) == 4


class TestPercentileMethod:
    def test_normal_fit_equals_classic_indices(self):
        x = np.random.default_rng(1).normal(10, 0.5, 120)
        spec = NonNormalSpec(usl=11.8, lsl=8.9)
        pi = percentile_indices(fit_distribution(x, Distribution.NORMAL), spec)
        m, s = x.mean(), x.std(ddof=1)
        assert pi.pp == pytest.approx((11.8 - 8.9) / (6 * s), rel=1e-9)
        assert pi.ppk == pytest.approx(min(11.8 - m, m - 8.9) / (3 * s), rel=1e-9)
        assert normal_theory_ppk(x, spec) == pytest.approx(pi.ppk, rel=1e-12)

    def test_analytic_lognormal_percentiles(self):
        # For a known lognormal, X_p = exp(mu + sigma z_p).
        x = np.random.default_rng(3).lognormal(0.2, 0.3, 500)
        f = fit_distribution(x, Distribution.LOGNORMAL)
        mu, sg = f.params["mu (log)"], f.params["sigma (log)"]
        pi = percentile_indices(f, NonNormalSpec(usl=4.0, lsl=0.3))
        z = stats.norm.ppf(P_HIGH)
        assert pi.x_high == pytest.approx(np.exp(mu + sg * z), rel=1e-12)
        assert pi.x_low == pytest.approx(np.exp(mu - sg * z), rel=1e-12)
        assert pi.x_mid == pytest.approx(np.exp(mu), rel=1e-12)
        assert pi.ppm_above == pytest.approx(1e6 * stats.norm.sf((np.log(4.0) - mu) / sg), rel=1e-12)

    def test_boundary_makes_it_one_sided(self):
        x = np.random.default_rng(4).weibull(1.5, 200) * 2
        f = fit_distribution(x, Distribution.WEIBULL)
        pi = percentile_indices(f, NonNormalSpec(usl=8.0, lsl=0.0, lsl_is_boundary=True))
        assert pi.pp is None and pi.ppl is None and pi.ppm_below == 0.0 and pi.ppk == pi.ppu

    def test_monte_carlo_convergence(self):
        # True Ppk for Weibull(k=2, scale=1), USL = 3 (upper only).
        k, lam, usl = 2.0, 1.0, 3.0
        q = lambda p: lam * (-np.log(1 - p)) ** (1 / k)  # noqa: E731
        true_ppu = (usl - q(0.5)) / (q(P_HIGH) - q(0.5))
        x = np.random.default_rng(5).weibull(k, 50_000) * lam
        est = percentile_indices(fit_distribution(x, Distribution.WEIBULL), NonNormalSpec(usl=usl)).ppk
        assert est == pytest.approx(true_ppu, rel=0.01)

    def test_spec_validation(self):
        with pytest.raises(ValueError):
            NonNormalSpec()
        with pytest.raises(ValueError):
            NonNormalSpec(usl=1.0, lsl=2.0)
        with pytest.raises(ValueError):
            NonNormalSpec(lsl=0.0, lsl_is_boundary=True)


class TestBoxCox:
    def test_transform_is_increasing_for_negative_lambda(self):
        v = np.array([1.0, 2.0, 5.0])
        assert np.all(np.diff(boxcox_transform(v, -1.0)) > 0)
        assert boxcox_transform(np.e, 0.0) == pytest.approx(1.0)

    def test_scaled_transform_gives_same_indices_as_power_transform(self):
        x = np.random.default_rng(6).lognormal(1, 0.3, 150)
        r = box_cox(x, NonNormalSpec(usl=8.0, lsl=1.0), lam=0.5)
        y, u, l = np.sqrt(x), np.sqrt(8.0), np.sqrt(1.0)
        m, s = y.mean(), y.std(ddof=1)
        assert r.ppk == pytest.approx(min(u - m, m - l) / (3 * s), rel=1e-12)

    def test_lambda_options(self):
        x = np.random.default_rng(7).lognormal(1, 0.5, 200)
        spec = NonNormalSpec(usl=20.0)
        assert box_cox(x, spec, round_to_half=False).lambda_used == pytest.approx(box_cox(x, spec).lambda_mle)
        assert box_cox(x, spec, lam=0.0).lambda_used == 0.0

    def test_rejects_nonpositive_data_and_lsl(self):
        with pytest.raises(ValueError):
            box_cox(np.array([0.0, 1.0, 2.0]), NonNormalSpec(usl=3.0))
        with pytest.raises(ValueError, match="boundary"):
            box_cox(np.array([1.0, 2.0, 3.0, 4.0]), NonNormalSpec(usl=5.0, lsl=0.0))


@pytest.mark.filterwarnings("ignore::FutureWarning")  # scipy >= 1.17 p-value API change; statistic unchanged
def test_anderson_darling_matches_scipy_for_normal():
    x = np.random.default_rng(8).normal(0, 1, 80)
    m, s = x.mean(), x.std(ddof=1)
    assert anderson_darling(x, stats.norm(m, s).cdf) == pytest.approx(stats.anderson(x, "norm").statistic, rel=1e-9)
