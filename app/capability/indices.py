"""Capability / performance index formulas, sigma level, and Cpk CI.

All formulas, assumptions and references are documented in ``docs/FORMULAS.md``
§2.4–2.6. Design invariants:

* Cp-family functions take a ``sigma`` argument the caller has computed from
  *within* variation; Pp-family functions take a ``sigma`` from *overall*
  variation. The functions themselves are sigma-source agnostic by design, and
  the orchestrating :class:`CapabilityReport` builder wires the correct sigma to
  each family so they can never be crossed. A dedicated test asserts Pp/Ppk do
  not reuse the within sigma.
* One-sided specs: ``Cp``/``Pp`` are ``None`` when a bound is missing; ``Cpk``
  collapses to the single available one-sided index.
* Zero variation (``sigma == 0``): indices return ``+inf`` (defined sentinel),
  never raising ZeroDivisionError.
* Full float precision preserved; no rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite, isnan, nan, sqrt
from typing import Optional

from scipy.stats import norm

from app.data.models import SpecLimits

__all__ = [
    "IndexSet",
    "SigmaLevel",
    "CpkConfidenceInterval",
    "compute_index_set",
    "compute_sigma_level",
    "cpk_confidence_interval",
    "DEFAULT_CI_ALPHA",
]

DEFAULT_CI_ALPHA: float = 0.05  # -> 95% confidence interval


def _ratio(numerator: float, sigma: float, factor: float) -> float:
    """Return ``numerator / (factor * sigma)`` with the zero-variation sentinel.

    When ``sigma == 0`` the process has no spread, so any finite margin is
    infinitely many sigmas wide -> ``+inf`` (the documented sentinel). A
    non-positive numerator with zero sigma (mean exactly on/outside the bound)
    yields ``-inf`` or ``0`` naturally via the sign of the numerator.
    """
    if sigma == 0.0:
        if numerator > 0:
            return inf
        if numerator < 0:
            return -inf
        return 0.0
    return numerator / (factor * sigma)


@dataclass(frozen=True, slots=True)
class IndexSet:
    """A family of capability/performance indices computed from one sigma.

    Field names are generic (``cp``/``cpk``/...) but the same structure is used
    for both the Cp-family (within sigma) and the Pp-family (overall sigma); the
    ``family`` tag records which. ``cp`` and the paired ``cpk`` may be ``None``
    only for the potential index under a one-sided spec.
    """

    family: str  # "capability" (Cp*) or "performance" (Pp*)
    sigma: float
    mean: float
    cp: Optional[float]  # potential (None for one-sided spec)
    cpk: Optional[float]  # actual (always defined if any spec present)
    cpu: Optional[float]  # upper one-sided (None if no USL)
    cpl: Optional[float]  # lower one-sided (None if no LSL)

    def as_dict(self) -> dict[str, Optional[float] | str]:
        return {
            "family": self.family,
            "sigma": self.sigma,
            "mean": self.mean,
            "cp": self.cp,
            "cpk": self.cpk,
            "cpu": self.cpu,
            "cpl": self.cpl,
        }


def compute_index_set(
    *,
    mean: float,
    sigma: float,
    spec: SpecLimits,
    family: str,
) -> IndexSet:
    """Compute one index family (Cp* or Pp*) from a supplied sigma.

    Parameters
    ----------
    mean:
        Process location estimate (grand mean).
    sigma:
        The standard deviation to use. The caller is responsible for passing the
        correct sigma for ``family`` (within for capability, overall for
        performance).
    spec:
        Specification limits (one- or two-sided).
    family:
        ``"capability"`` or ``"performance"`` — recorded on the result.

    Notes
    -----
    * Two-sided: ``cp`` and the min-based ``cpk`` are both defined.
    * One-sided upper: ``cp = None``, ``cpk = cpu``.
    * One-sided lower: ``cp = None``, ``cpk = cpl``.
    * ``sigma == 0``: ``+inf`` sentinels via :func:`_ratio`.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}.")

    usl = spec.usl
    lsl = spec.lsl

    cpu = _ratio(usl - mean, sigma, 3.0) if usl is not None else None
    cpl = _ratio(mean - lsl, sigma, 3.0) if lsl is not None else None

    if usl is not None and lsl is not None:
        cp = _ratio(usl - lsl, sigma, 6.0)
        # cpk = min(cpu, cpl); both are finite or +/-inf sentinels.
        cpk = min(cpu, cpl)  # type: ignore[type-var]
    elif usl is not None:
        cp = None
        cpk = cpu
    elif lsl is not None:
        cp = None
        cpk = cpl
    else:  # pragma: no cover - SpecLimits forbids this
        raise ValueError("SpecLimits must define at least one bound.")

    return IndexSet(
        family=family,
        sigma=float(sigma),
        mean=float(mean),
        cp=cp,
        cpk=cpk,
        cpu=cpu,
        cpl=cpl,
    )


@dataclass(frozen=True, slots=True)
class SigmaLevel:
    """Process sigma level (Z-bench) and expected nonconforming fraction.

    Attributes
    ----------
    z_usl, z_lsl:
        One-sided Z distances to each bound (``None`` if that bound absent).
        Under zero variation these are ``+inf`` (mean strictly conforming on
        that side), ``-inf`` (strictly nonconforming), or ``NaN`` (mean exactly
        on the bound, where Z = 0/0 is undefined).
    p_above, p_below:
        Expected fraction beyond USL / below LSL under the normal model.
    p_total:
        Total expected nonconforming fraction. ``NaN`` when undefined (zero
        variation with the mean exactly on a specification bound).
    z_bench:
        The process sigma level, ``Phi^{-1}(1 - p_total)``. ``+inf`` when
        ``p_total`` underflows to 0 (extremely capable / zero-variation with
        the mean inside spec); ``-inf`` when the process is wholly
        nonconforming; ``NaN`` when undefined (see ``p_total``).
    dpmo:
        Defects per million opportunities (``p_total * 1e6``); ``NaN`` when
        ``p_total`` is undefined.
    with_shift:
        If a 1.5-sigma shift convention was applied (off by default).
    """

    z_usl: Optional[float]
    z_lsl: Optional[float]
    p_above: float
    p_below: float
    p_total: float
    z_bench: float
    dpmo: float
    with_shift: bool = False

    def as_dict(self) -> dict[str, Optional[float] | bool]:
        return {
            "z_usl": self.z_usl,
            "z_lsl": self.z_lsl,
            "p_above": self.p_above,
            "p_below": self.p_below,
            "p_total": self.p_total,
            "z_bench": self.z_bench,
            "dpmo": self.dpmo,
            "with_shift": self.with_shift,
        }


def compute_sigma_level(
    *,
    mean: float,
    sigma: float,
    spec: SpecLimits,
    apply_1p5_shift: bool = False,
) -> SigmaLevel:
    """Compute the process sigma level (Z-bench) under a normal model.

    See FORMULAS.md §2.5. ``apply_1p5_shift`` adds the Six Sigma 1.5-sigma
    convention to ``z_bench`` (off by default; flagged when on). Normality is
    assumed; when it is questionable the caller attaches a non-normality warning.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}.")

    usl = spec.usl
    lsl = spec.lsl

    if sigma == 0.0:
        # Degenerate (zero-spread) process: ALL mass sits exactly at `mean`.
        # The nonconforming fraction therefore depends entirely on where the
        # mean sits relative to each bound -- it is NOT automatically zero.
        #
        # Per bound, with all mass at `mean`:
        #   * mean strictly inside  -> that tail is empty      : p = 0, Z = +inf
        #   * mean strictly outside -> that tail holds it all  : p = 1, Z = -inf
        #   * mean exactly on bound -> Z = 0/0, genuinely undefined; the
        #     inclusive-limit reading gives p = 0 while the limiting-normal
        #     reading (sigma -> 0+) gives p = 0.5. Rather than silently pick
        #     one, we report NaN ("undefined"), which every exporter already
        #     renders as "NaN". The accompanying zero-variation reliability
        #     warning tells the engineer to verify the gauge.
        def _degenerate(bound: Optional[float], *, upper: bool) -> tuple[Optional[float], float]:
            """Return (Z, p) for one bound under a zero-spread distribution."""
            if bound is None:
                return None, 0.0
            margin = (bound - mean) if upper else (mean - bound)
            if margin > 0:      # strictly conforming on this side
                return inf, 0.0
            if margin < 0:      # strictly nonconforming on this side
                return -inf, 1.0
            return nan, nan     # exactly on the bound -> undefined

        z_usl, p_above = _degenerate(usl, upper=True)
        z_lsl, p_below = _degenerate(lsl, upper=False)
        p_total = p_above + p_below  # NaN propagates if either side is on-bound

        if isnan(p_total):
            z_bench = nan
        elif p_total <= 0.0:
            z_bench = inf
        elif p_total >= 1.0:
            z_bench = -inf
        else:  # pragma: no cover - unreachable: a valid spec has USL > LSL,
            # so at most one side can be violated and p_total is 0, 1 or NaN.
            z_bench = float(norm.ppf(1.0 - p_total))

        # `isfinite` is False for both inf and NaN, so the 1.5-sigma shift is
        # correctly skipped here; the flag stays truthful either way.
        if apply_1p5_shift and isfinite(z_bench):
            z_bench = z_bench + 1.5

        return SigmaLevel(
            z_usl=z_usl,
            z_lsl=z_lsl,
            p_above=p_above,
            p_below=p_below,
            p_total=p_total,
            z_bench=z_bench,
            dpmo=float(p_total * 1e6),
            with_shift=apply_1p5_shift,
        )

    z_usl = (usl - mean) / sigma if usl is not None else None
    z_lsl = (mean - lsl) / sigma if lsl is not None else None
    p_above = float(norm.sf(z_usl)) if z_usl is not None else 0.0
    p_below = float(norm.sf(z_lsl)) if z_lsl is not None else 0.0
    p_total = p_above + p_below

    if p_total <= 0.0:
        z_bench = inf
    elif p_total >= 1.0:
        z_bench = -inf
    else:
        z_bench = float(norm.ppf(1.0 - p_total))

    if apply_1p5_shift and isfinite(z_bench):
        z_bench = z_bench + 1.5

    return SigmaLevel(
        z_usl=z_usl,
        z_lsl=z_lsl,
        p_above=p_above,
        p_below=p_below,
        p_total=p_total,
        z_bench=z_bench,
        dpmo=float(p_total * 1e6),
        with_shift=apply_1p5_shift,
    )


@dataclass(frozen=True, slots=True)
class CpkConfidenceInterval:
    """Approximate confidence interval for Cpk (Bissell 1990; Montgomery 8.24).

    Attributes
    ----------
    cpk:
        The point estimate the interval is built around.
    lower, upper:
        Interval bounds.
    confidence:
        Confidence level (e.g. 0.95).
    n:
        Number of individual observations used.
    approximate:
        Always ``True`` — this is a large-sample normal approximation, never
        presented as exact.
    caveats:
        Human-readable caveats (small N, within-sigma dof mismatch,
        non-normality, instability) attached by the caller/report builder.
    """

    cpk: float
    lower: float
    upper: float
    confidence: float
    n: int
    approximate: bool = True
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def as_dict(self) -> dict[str, object]:
        return {
            "cpk": self.cpk,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "n": self.n,
            "approximate": self.approximate,
            "caveats": list(self.caveats),
        }


def cpk_confidence_interval(
    cpk: float,
    n: int,
    *,
    alpha: float = DEFAULT_CI_ALPHA,
    extra_caveats: tuple[str, ...] = (),
) -> CpkConfidenceInterval:
    """Approximate ``100(1-alpha)%`` CI for Cpk.

    ``CI = cpk ± z_{alpha/2} * sqrt( 1/(9N) + cpk^2 / (2(N-1)) )`` — Bissell
    (1990), Montgomery eq. 8.24. See FORMULAS.md §2.6 for assumptions and
    limitations. Default ``alpha = 0.05`` (95%).

    Raises
    ------
    ValueError
        If ``n < 2`` (variance term undefined), ``alpha`` not in (0, 1), or
        ``cpk`` is not finite (e.g. the zero-variation ``+inf`` sentinel — a CI
        is meaningless there and the caller should skip it).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    if n < 2:
        raise ValueError(
            f"Cpk CI requires n >= 2 (variance term uses N-1); got n = {n}."
        )
    if not isfinite(cpk):
        raise ValueError(
            "Cpk CI is undefined for a non-finite Cpk (e.g. zero-variation "
            "+inf sentinel); skip the interval and report the zero-variation "
            "warning instead."
        )

    z = float(norm.ppf(1.0 - alpha / 2.0))
    se = sqrt(1.0 / (9.0 * n) + (cpk * cpk) / (2.0 * (n - 1)))
    margin = z * se

    caveats: list[str] = []
    if n < 30:
        caveats.append(
            f"Small sample (N={n} < 30): this asymptotic interval is only a "
            "rough approximation and may be too narrow."
        )
    caveats.extend(extra_caveats)

    return CpkConfidenceInterval(
        cpk=float(cpk),
        lower=float(cpk - margin),
        upper=float(cpk + margin),
        confidence=float(1.0 - alpha),
        n=int(n),
        approximate=True,
        caveats=tuple(caveats),
    )
