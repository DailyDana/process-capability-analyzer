# Statistical Methodology Reference — SPC / Process Capability Analyzer

This document is the **single authoritative reference** for every statistical
quantity the application computes: formula, definition, interpretation,
assumptions, estimator conventions, edge-case behavior, and literature source.
Nothing is computed in code that is not documented here. Where more than one
accepted SPC convention exists, a **statistical-convention flag** records the
choice explicitly instead of choosing silently.

**Contents**

1. [Descriptive statistics and control-chart constants](#1-descriptive-statistics-and-control-chart-constants)
   — mean, median, variance/std dev, range; d2, d3, c4, A2, A3, D3, D4, B3, B4,
   E2; table-vs-exact convention; data-regime guardrail.
2. [Variation estimation, capability indices, sigma level, reliability](#2-variation-estimation-capability-indices-sigma-level-reliability)
   — sigma_within (R̄/d2, S̄/c4, pooled, MR̄/d2) vs sigma_overall; Cp, Cpk, Cpu,
   Cpl, Pp, Ppk, Ppu, Ppl; Z, Z-bench, DPMO, 1.5σ shift; Cpk confidence
   interval (Bissell); reliability guards; normality testing.
3. [Control-limit calculators and the WE/Nelson rule engine](#3-control-limit-calculators-and-the-wenelson-rule-engine)
   — X̄–R, X̄–S, I–MR limits; zones; Western Electric rules; Nelson rules;
   rule-set differences; missing/insufficient-data handling.
4. [Chart construction and rendering](#4-chart-construction-and-rendering-presentation-only)
   — presentation-layer invariants (no statistical content).
5. [Interpretation engine and report serialization](#5-interpretation-engine-and-report-serialization)
   — capability bands and all configurable engineering thresholds (conventions,
   not laws); deterministic rule triggers; export guarantees.

## Authoritative references (used throughout)

- **[M]** Montgomery, D.C. (2013). *Introduction to Statistical Quality
  Control*, 7th ed., Wiley. (Chapters 3, 6; Appendix Table VI.)
- **[AIAG]** Automotive Industry Action Group (2005). *Statistical Process
  Control (SPC) Reference Manual*, 2nd ed.
- **[NIST]** NIST/SEMATECH *e-Handbook of Statistical Methods*, Section 6
  (Process or Product Monitoring and Control). https://www.itl.nist.gov/div898/handbook/
- **[David]** David, H.A. & Nagaraja, H.N. (2003). *Order Statistics*, 3rd ed.,
  Wiley. (Distribution of the sample range.)
- **[ASTM]** ASTM International. *Manual on Presentation of Data and Control
  Chart Analysis* (STP-15D).

A **statistical-convention flag** in the text marks any place where more than
one defensible convention exists; the tool documents the choice rather than
making it silently.

---

## 1. Descriptive statistics and control-chart constants

### 1.1 Notation

| Symbol | Meaning |
|---|---|
| `x_i` | the i-th individual measurement |
| `N` | total number of individual measurements |
| `n` | subgroup size (measurements per subgroup) |
| `k` | number of subgroups |
| `x-bar` | sample mean of a series |
| `x-double-bar` | grand mean (mean of subgroup means) |
| `s` | sample standard deviation |
| `R` | range (max − min) |
| `sigma` | true process standard deviation (estimated, never known) |
| `Phi`, `phi` | standard-normal CDF and PDF |
| `Gamma` | the gamma function |

---

### 1.2 Mean (arithmetic average)

- **Formula:** `x-bar = (1/N) * sum_{i=1..N} x_i`
- **Meaning:** central location of the measurements; estimator of the process
  mean `mu`.
- **Assumptions:** none beyond a meaningful interval scale. Sensitive to
  outliers (unlike the median).
- **Source:** [M] Ch. 3; [NIST] 1.3.5.1.

### 1.3 Median

- **Formula:** the 50th percentile. For sorted `x_(1) <= ... <= x_(N)`:
  - odd `N`: `x_((N+1)/2)`
  - even `N`: `(x_(N/2) + x_(N/2 + 1)) / 2` (linear interpolation).
- **Meaning:** outlier-robust location estimate. Comparing mean vs. median is a
  quick, informal skewness check used later in interpretation.
- **Assumptions:** ordinal scale sufficient.
- **Source:** [NIST] 1.3.5.1.

### 1.4 Variance and standard deviation

- **Sample (default, `ddof = 1`, Bessel-corrected):**
  - `s^2 = (1/(N-1)) * sum (x_i - x-bar)^2`
  - `s = sqrt(s^2)`
- **Population (`ddof = 0`):** divide by `N` instead of `N-1`.
- **Meaning:** spread of the data. `s` (overall, computed on all individuals) is
  the basis of **overall sigma** used for Pp/Ppk (see §2.3).
- **Assumptions:** the sample estimator is unbiased for the *variance*; note that
  `s` itself is a **biased** estimator of `sigma` (corrected by `c4`, §1.6).
- **Statistical-convention flag:** SPC analyses samples of an ongoing process,
  so `ddof = 1` is the default. `ddof = 0` is exposed for completeness.
- **Source:** [M] Ch. 3; [NIST] 1.3.5.6.

### 1.5 Minimum, maximum, range

- **Formulas:** `min = min_i x_i`, `max = max_i x_i`, `R = max − min`.
- **Meaning:** `R` is the dispersion measure underlying the R-chart and the
  `R-bar / d2` estimate of within-subgroup sigma (see §2.2 and §3.2).
- **Assumptions:** `R` uses only two order statistics, so it is less efficient
  than `s` for large `n` — the reason charts switch from R to S at large `n`.
- **Source:** [M] Ch. 6.

---

### 1.6 Control-chart constants

These constants convert subgroup ranges / standard deviations into unbiased
sigma estimates and 3-sigma control limits. The application stores the
**canonical published values** (below) *and* computes them from first
principles; unit tests assert the two agree, validating both.

#### 1.6.1 `c4` — bias correction for the sample standard deviation

- **Formula:** `c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)`
- **Meaning:** `E[s] = c4 * sigma`. Because `s` under-estimates `sigma`, an
  unbiased estimate is `sigma-hat = s-bar / c4` (used by the S-chart and the
  S-bar/c4 within-sigma option).
- **Assumptions:** the underlying data are **normally distributed**.
- **Implementation note:** evaluated in log-gamma space for numerical stability.
- **Source:** [M] Appendix VI; [NIST] 6.3.2.

#### 1.6.2 `d2`, `d3` — mean and standard deviation of the relative range

- **Definitions:** for `W = R / sigma` of a normal sample of size `n`:
  - `d2 = E[W]`
  - `d3 = sqrt(Var[W])`
- **Meaning:** `sigma-hat = R-bar / d2` is the classic within-subgroup sigma
  estimate; `d3` sets the spread of the R-chart limits.
- **Computation:** the range CDF is
  `P(W <= w) = n * integral_{-inf}^{inf} phi(x) [Phi(x+w) − Phi(x)]^(n-1) dx`,
  and moments use the survival form `E[W^k] = integral_0^inf k w^(k-1) (1 − P(W<=w)) dw`.
  Evaluated numerically (scipy.integrate.quad).
- **Assumptions:** normality; i.i.d. within subgroup.
- **Source:** [David] Ch. 3 (range distribution); [M] Appendix VI.

#### 1.6.3 Chart factors derived from `d2`, `d3`, `c4`

| Factor | Formula | Used by |
|---|---|---|
| `A2` | `3 / (d2 * sqrt(n))` | X-bar limits from `R-bar`: `x-double-bar ± A2 * R-bar` |
| `A3` | `3 / (c4 * sqrt(n))` | X-bar limits from `S-bar`: `x-double-bar ± A3 * S-bar` |
| `D3` | `max(0, 1 − 3 * d3 / d2)` | R-chart LCL factor: `LCL = D3 * R-bar` |
| `D4` | `1 + 3 * d3 / d2` | R-chart UCL factor: `UCL = D4 * R-bar` |
| `B3` | `max(0, 1 − (3/c4) * sqrt(1 − c4^2))` | S-chart LCL factor: `LCL = B3 * S-bar` |
| `B4` | `1 + (3/c4) * sqrt(1 − c4^2)` | S-chart UCL factor: `UCL = B4 * S-bar` |
| `E2` | `3 / d2` (with `n = 2`, `E2 = 2.660`) | I-chart limits from `MR-bar`: `x-bar ± E2 * MR-bar` |

> **Statistical-convention flag (E2 rounding).** The widely quoted `E2 = 2.660`
> for `n = 2` is `3 / 1.128`, i.e. computed from the *rounded* `d2 = 1.128`. The
> exact value is `3 / d2(2) = 3 / 1.128379 = 2.6587`. The tool computes the
> exact form and treats `2.660` as a display-rounded landmark; the difference
> (~0.05%) is immaterial to control limits but is documented rather than hidden.

- **Meaning:** all encode Shewhart's 3-sigma limits, expressed in units of the
  measured dispersion statistic so the analyst multiplies a single factor.
- **Assumptions:** normality; rational subgroups (within-subgroup variation is
  common-cause only). `D3 = 0` for `n <= 6` because the lower 3-sigma bound on
  the range is negative and truncated to zero.
- **Source:** [M] Appendix VI, eqs. for A2/A3/D3/D4/B3/B4/E2; [ASTM] STP-15D.

---

### 1.7 Data-regime guardrail (definitions fixed in M1)

The capability layer must **not** treat Cp/Cpk and Pp/Ppk as interchangeable.
The `DataRegime` enum encodes when each is valid:

| Regime | Structure | Charts | Within-sigma (Cp/Cpk) | Overall-sigma (Pp/Ppk) |
|---|---|---|---|---|
| Subgrouped | rational subgroups `n >= 2` | X-bar/R or X-bar/S | `R-bar/d2` or `S-bar/c4` | `s` over all data |
| Individuals | `n = 1` | I-MR | `MR-bar/d2`, `d2 = 1.128` | `s` over all data |
| Unordered | no time/subgroup order | none valid | **suppressed** | `s` over all data |

- **Rationale:** within-subgroup sigma estimates *common-cause* variation only.
  Without rational subgroups (Unordered), no such estimate is defensible, so
  Cp/Cpk are suppressed and only Pp/Ppk are reported. The numeric formulas for
  these indices are documented in §2.4.
- **How the regime is determined (enforcement, added in the hardening
  milestone):** the regime is set from an explicit **user attestation** that the
  measurements are in production/time order (`AnalysisRequest.time_ordered`,
  surfaced in the UI as *"Measurements are in production/time order"*, default
  on). It is **never inferred** from the data: a sortedness or autocorrelation
  heuristic can be defeated by ordinary process behaviour in both directions
  (a genuinely sorted production run vs. a coincidentally monotone sample), so
  guessing would produce false confidence in exactly the place this guardrail
  exists to prevent it.
- **Enforcement point:** `app/services/analysis.run_analysis` — the service
  layer, not the UI. When the attestation is absent the regime is `UNORDERED`
  and the service passes `sigma_within=None` and `stability=None` into the
  existing `build_capability_report`, which is the single place that binds a
  sigma to an index family. Consequences:
  - `capability` (Cp/Cpk/Cpu/Cpl), the Cpk confidence interval and the
    within-sigma level are **not computed** and render as `n/a` everywhere;
  - `performance` (Pp/Ppk/Ppu/Ppl) is **unchanged and fully valid**, since it
    depends only on the overall sigma;
  - the control charts are still drawn but carry **no stability verdict**
    (without a known time order, run rules have no temporal meaning);
  - the interpretation engine emits `within_capability_not_assessed` and
    cannot emit any Cpk-band conclusion, so the process is never called
    "capable" on the strength of an index that was not computed.
  No formula is duplicated or moved into the service/UI layers; the guardrail
  only decides *which already-validated inputs are supplied*.
- **Source:** [AIAG] SPC manual (short-term vs. long-term study distinction);
  [M] Ch. 8.

---

## 2. Variation estimation, capability indices, sigma level, reliability

### 2.1 The central distinction: within-subgroup vs overall variation

This is the single most important correctness requirement of the whole tool.
**Cp/Cpk use within-subgroup (short-term) sigma; Pp/Ppk use overall (long-term)
sigma. They are never interchangeable and the two sigmas are estimated by
entirely separate code paths.**

| Quantity | Symbol | Estimates | Feeds |
|---|---|---|---|
| Within-subgroup sigma | `sigma_within` | inherent common-cause variation *if the process held still* | **Cp, Cpu, Cpl, Cpk** |
| Overall sigma | `sigma_overall` | total variation actually experienced, incl. drift/shift between subgroups | **Pp, Ppu, Ppl, Ppk** |

If the process is perfectly stable the two converge. The gap between them is a
direct, quantitative signature of instability. Interpretation of that gap is an
matter for the interpretation engine (§5); this section only guarantees the
two numbers are computed correctly and independently.

- **Source:** [AIAG] SPC manual, Ch. on capability vs performance; [M] Ch. 8.4;
  [NIST] 6.1.6, 6.4.

### 2.2 Within-subgroup sigma estimators

Let there be `k` subgroups. `R_j`, `s_j`, `n_j` are the range, sample standard
deviation, and size of subgroup `j`. Constants `d2(n)`, `c4(n)` are from M1.

**(a) Average range — `R-bar / d2` (default for subgroups)**

- `R-bar = (1/k) * sum_j R_j`
- `sigma_within = R-bar / d2(n)`
- **Requires** equal subgroup size `n` (d2 depends on n). Classic and matches
  the X-bar/R chart. Default estimator per project decision.
- **Assumptions:** normality; rational subgroups; balanced `n`.
- **Source:** [M] eq. 6.5; [AIAG]; [NIST] 6.3.2.

**(b) Average standard deviation — `S-bar / c4` (auto-selected for `n >= 9`)**

- `S-bar = (1/k) * sum_j s_j`   (each `s_j` uses `ddof = 1`)
- `sigma_within = S-bar / c4(n)`
- **Rationale for the `n >= 9` switch:** the range uses only two order
  statistics, so its statistical efficiency as an estimator of `sigma` falls as
  `n` grows (approx. 0.85 at `n = 10` and worse beyond), whereas `s` uses all
  observations. The classic guidance (Montgomery §6.3.1; AIAG) is to move from
  the R-chart to the S-chart for larger subgroups; we mirror that for the sigma
  estimate and switch the default to `S-bar/c4` at `n >= 9`. The choice is
  always exposed and recorded in the result.
- **Assumptions:** normality; rational subgroups; balanced `n`.
- **Source:** [M] §6.3.1; [NIST] 6.3.2.

**(c) Pooled standard deviation (used for unbalanced subgroups)**

- `s_p = sqrt( sum_j (n_j - 1) * s_j^2  /  sum_j (n_j - 1) )`
- Unbiased estimate: `sigma_within = s_p / c4(nu + 1)`, where
  `nu = sum_j (n_j - 1)` is the pooled degrees of freedom and `c4` is evaluated
  at the equivalent sample size `nu + 1` (so that its internal `nu = (nu+1)-1`).
- **Meaning:** the maximum-likelihood-style pooling of within-subgroup variance;
  the only within estimator that correctly handles **unequal** subgroup sizes.
- **Assumptions:** normality; rational subgroups. Handles unbalanced `n`.
- **Source:** [M] §6.3.2 (pooled estimator and c4 bias correction expressed via
  degrees of freedom).

**(d) Average moving range — `MR-bar / d2(2)` (individuals, `n = 1`)**

- `MR_i = |x_i - x_{i-1}|` for `i = 2..N`;  `MR-bar = mean(MR_i)`
- `sigma_within = MR-bar / d2(2) = MR-bar / 1.128`
- **Meaning:** the only within estimator available when there is one reading per
  time point (I-MR chart). Uses the moving range of consecutive pairs, so the
  relevant subgroup size is 2 and `d2(2) = 1.128`.
- **Assumptions:** normality; observations time-ordered; adjacent readings are
  the rational "subgroup". Sensitive to autocorrelation.
- **Source:** [M] §6.4; [AIAG]; [NIST] 6.3.2.

> **Statistical-convention flag (estimator choice).** For balanced subgroups the
> three subgroup estimators (a)/(b)/(c) agree closely under normality but are not
> numerically identical in finite samples. The tool defaults to (a), auto-selects
> (b) at `n >= 9`, uses (c) only when sizes are unequal, and always records which
> estimator produced the reported `sigma_within`.

> **Statistical-convention flag (table vs exact constants).** The R-bar/d2,
> S-bar/c4, and pooled estimators use the **published tabulated** constants
> (`d2`, `c4`) for `2 <= n <= 25`, matching textbook worked examples and
> commercial SPC software so hand-checks agree. For `n > 25` (beyond the printed
> table) the tool falls back to the exact integral/gamma-function values from M1.
> The difference between tabulated and exact is ~1e-3 and immaterial, but the
> convention is fixed and documented rather than chosen silently.

### 2.3 Overall (long-term) sigma

- `sigma_overall = s = sqrt( (1/(N-1)) * sum_{i=1..N} (x_i - x-bar)^2 )`
  over **all** `N` individual observations pooled together (ignoring subgroup
  membership), using Bessel's correction (`ddof = 1`).
- **Critical implementation guarantee:** Pp/Ppk are computed from this and only
  this. They must never reuse any within-subgroup sigma. This is enforced by
  separate functions and asserted by a dedicated test.
- **Assumptions:** the `N` observations are a representative sample of the total
  process output. `s` is a slightly biased estimator of `sigma`; the bias is not
  removed here because Pp/Ppk are conventionally defined on `s` itself
  ([AIAG]).
- **Source:** [AIAG] performance-index definition; [M] §8.4; [NIST] 6.1.6.

### 2.4 Capability and performance indices

Let `mu-hat = x-double-bar` (grand mean of all observations). Choose sigma per
the index family (within for Cp*, overall for Pp*).

**Two-sided (both USL and LSL present):**

```
Cp   = (USL - LSL) / (6 * sigma_within)     # potential capability, centering ignored
Cpu  = (USL - mu-hat) / (3 * sigma_within)  # upper capability
Cpl  = (mu-hat - LSL) / (3 * sigma_within)  # lower capability
Cpk  = min(Cpu, Cpl)                        # actual capability, penalizes off-center

Pp   = (USL - LSL) / (6 * sigma_overall)    # potential performance
Ppu  = (USL - mu-hat) / (3 * sigma_overall)
Ppl  = (mu-hat - LSL) / (3 * sigma_overall)
Ppk  = min(Ppu, Ppl)                        # actual performance
```

- **Cp / Pp** measure *potential* — the spec width relative to `6 sigma`,
  assuming the process could be perfectly centered. They ignore where the mean
  actually sits.
- **Cpk / Ppk** account for the actual centering by taking the worse of the two
  one-sided margins. Consequently `Cpk <= Cp` always, with equality **iff** the
  process is exactly centered (`mu = (USL+LSL)/2`). The analogous relation holds
  for Ppk/Pp.
- **Source:** [M] §8.3–8.4; [AIAG]; [NIST] 6.1.6.

**One-sided specifications:**

- **Upper spec only (USL, no LSL):** `Cp`/`Pp` are undefined (returned as
  `None`); `Cpk = Cpu`, `Ppk = Ppu`.
- **Lower spec only (LSL, no USL):** `Cp`/`Pp` undefined; `Cpk = Cpl`,
  `Ppk = Ppl`.
- **Source:** [AIAG] one-sided capability; [NIST] 6.1.6.

**Zero variation (`sigma = 0`):**

- Every index has `sigma` in the denominator, so `sigma = 0` makes them
  mathematically `+inf` (spec width is infinitely many sigmas wide). The tool
  returns `+inf` as an explicit, defined sentinel (never a divide-by-zero
  crash) and attaches a reliability warning: zero observed variation is almost
  always a measurement-resolution or data-entry artifact, not a real process.

**Zero variation and the sigma level / DPMO (mean position matters):**

Under `sigma = 0` all probability mass sits exactly at the mean, so the
nonconforming fraction is **not** automatically zero — it depends entirely on
where the mean sits relative to each bound. Per bound:

| Mean position | Z for that bound | p for that tail |
|---|---|---|
| strictly inside the bound | `+inf` | `0` |
| strictly outside the bound | `-inf` | `1` |
| exactly **on** the bound | `NaN` (undefined) | `NaN` |

- Strictly inside both bounds: `p_total = 0`, `DPMO = 0`, `Z_bench = +inf`.
- Strictly outside a bound: `p_total = 1`, `DPMO = 1e6`, `Z_bench = -inf`
  (a process producing 100 % nonconforming parts).
- Exactly on a bound: `Z = 0/0` is **genuinely undefined**. The inclusive-limit
  reading gives `p = 0`; the limiting-normal reading (`sigma -> 0+`) gives
  `p = 0.5`. Rather than silently adopting one convention, the tool reports
  `NaN` ("undefined"), which every exporter renders as `NaN`, and the
  zero-variation reliability warning tells the engineer to verify the gauge.
- **Consistency requirement:** this makes the sigma level agree with the index
  family, which already returns `Cpk = 0` on a bound and `-inf` outside it. A
  report can never simultaneously claim `Cpk <= 0` and `DPMO = 0`/`Z = +inf`.
- **Source:** engineering convention; documented behavior, not a published
  formula.

### 2.5 Process sigma level (Z-bench) and nonconforming fraction

Under a normal model with the chosen sigma:

```
Z_USL = (USL - mu-hat) / sigma        # if USL present
Z_LSL = (mu-hat - LSL) / sigma        # if LSL present
p_above = Phi(-Z_USL)                 # expected fraction above USL
p_below = Phi(-Z_LSL)                 # expected fraction below LSL
p_total = p_above + p_below           # total expected nonconforming fraction
Z_bench = Phi^{-1}(1 - p_total)       # the "process sigma level" (Z-bench)
DPMO    = p_total * 1e6               # defects per million opportunities
```

- **Meaning:** `Z_bench` expresses total nonconformance as a single equivalent
  number of standard deviations to the nearer effective spec — the metric behind
  "process sigma level". Computed separately for `sigma_within`
  (short-term sigma level) and `sigma_overall` (long-term).
- **Relationship:** for a two-sided, centered process `Z_bench ≈ 3 * Cpk`; in
  general the exact `Z_bench` from `p_total` is reported rather than the `3*Cpk`
  shortcut.
- **1.5-sigma shift:** the Six Sigma convention of adding `1.5` to a short-term
  Z to obtain a "long-term" sigma level is a **convention, not a derived
  result**. It is **off by default**, exposed as an explicit option, and always
  labeled as a convention when applied.
- **Assumptions:** normality; a stable process. If normality is questionable the
  `p_total`/`Z_bench`/`DPMO` translation is unreliable and the tool says so
  (see 2.7).
- **Source:** [M] §8.3.2; [AIAG]; [NIST] 6.1.6, 6.5.

### 2.6 Confidence interval for Cpk

- **Method:** large-sample normal-approximation interval (Bissell 1990;
  Montgomery §8.3.1, eq. 8.24). The approximate `100(1 - alpha)%` interval is

  ```
  Cpk_hat  ±  z_{alpha/2} * sqrt( 1/(9N) + Cpk_hat^2 / (2 (N - 1)) )
  ```

  equivalently written `Cpk_hat [ 1 ± z_{alpha/2} sqrt( 1/(9 N Cpk_hat^2) + 1/(2(N-1)) ) ]`.
  `N` is the total number of individual observations; `z_{alpha/2}` is the
  standard-normal quantile (`z_{0.025} = 1.959964` for the default 95%).

- **Confidence level:** 95% by default (`alpha = 0.05`), configurable.

- **Assumptions:**
  1. The individual measurements are approximately **normally distributed**.
  2. The process is **stable / in statistical control**.
  3. `N` is **large** (the interval is asymptotic; it degrades for small `N`).
  4. `sigma` is estimated with `N - 1` degrees of freedom (the derivation
     assumes the overall sample standard deviation).

- **Limitations (surfaced in the result, never hidden):**
  - It is an **approximate**, not exact, interval. The result is flagged
    `approximate = True` and is explicitly **not** presented as exact.
  - When `Cpk` is computed from a **within-subgroup** sigma (`R-bar/d2`,
    `S-bar/c4`, `MR-bar/d2`), the effective degrees of freedom differ from
    `N - 1`, so the interval is only a first-order approximation; this caveat is
    attached to the result.
  - If normality is questionable (see 2.7) or the process is unstable, the
    interval's assumptions are violated; the result carries an explicit note and
    must not be read as a valid normal-theory interval.
  - For very small `N` the lower bound can fall below 0 or become meaningless;
    reported as-is with the small-sample warning attached.

- **Source:** Bissell, A.F. (1990), "How reliable is your capability index?",
  *Applied Statistics* 39(3):331–340; [M] §8.3.1 eq. 8.24; [Kotz & Lovelace,
  *Process Capability Indices in Theory and Practice*, 1998].

### 2.7 Reliability guards (warnings, never hard failures)

The following conditions attach a structured warning to the capability result
but **do not** suppress or refuse the indices (per project decision):

| Guard | Condition | Rationale |
|---|---|---|
| Small sample | total `N < 30` individual observations | capability indices have wide sampling variability at low `N` ([M] §8.7). |
| Few subgroups | number of rational subgroups `< 20` | too few subgroups to trust the within-subgroup sigma / stability picture ([AIAG]). |
| Zero variation | `sigma = 0` | indices are `+inf`; almost always a resolution/data artifact. |
| Non-normality | normality test rejects at `alpha = 0.05` | Cp/Cpk-to-PPM translation and the Cpk CI assume normality; if rejected these are unreliable. Detection only in v1 (no transformation). |
| Stability unknown | stability not yet assessed | when no SPC rule evaluation has been run, capability carries a "stability not verified" note; capability on an unstable process is not trustworthy. |

- **Normality test:** the **Shapiro–Wilk** test (`scipy.stats.shapiro`) is used
  as the primary check (good power for small-to-moderate `N`); the
  **Anderson–Darling** statistic is also reported. Rejection at `alpha = 0.05`
  raises the non-normality warning. Only *detection and warning* are implemented
  in v1 — no Box–Cox/Weibull transformation (architecture left extensible).
- **Source:** [M] §8.7 (reliability of capability estimates); [AIAG] (minimum
  data guidance, typically >= 25 subgroups / >= 100 measurements for a capability
  study — our defaults of N>=30 and >=20 subgroups are conservative warning
  thresholds, documented as such and configurable).

---

## 3. Control-limit calculators and the WE/Nelson rule engine

### 3.1 Control charts vs capability — separate sigma, separate purpose

Control charts answer a **different question** from capability and use their own
limit methodology. They must not reuse the capability-layer sigma machinery
blindly:

| | Control charts (§3) | Capability (§2) |
|---|---|---|
| Question | *Is the process stable/predictable?* | *Can the stable process meet spec?* |
| Limits from | subgroup dispersion statistic (`R̄`, `S̄`, `MR̄`) via chart constants | spec limits vs `6σ` |
| Center line | process's own average (`x-double-bar`, `R̄`, ...) | n/a |
| Spec limits | **never** plotted on control charts | the whole point |

Control limits are computed directly from the plotted statistic's own average
(e.g. the X̄ chart's limits come from `R̄` or `S̄`, never from a capability-layer
sigma). The within-subgroup constants are the validated table values of §1.6.

- **Source:** [M] Ch. 6; [AIAG]; [NIST] 6.3.

### 3.2 X̄–R chart (subgroups, typically `2 <= n <= ~8`)

Let `k` subgroups each of size `n`. `x̄_j`, `R_j` are the mean and range of
subgroup `j`.

```
x-double-bar = (1/k) * sum_j x̄_j        # grand mean  -> X̄-chart CL
R̄            = (1/k) * sum_j R_j         # mean range  -> R-chart CL

X̄ chart:  UCL = x-double-bar + A2 * R̄
          CL  = x-double-bar
          LCL = x-double-bar - A2 * R̄

R chart:  UCL = D4 * R̄
          CL  = R̄
          LCL = D3 * R̄        (= 0 for n <= 6, since D3 = 0)
```

- **Constants:** `A2`, `D3`, `D4` from the M1 table at subgroup size `n`.
- **Assumptions:** rational subgroups; approx. normal; balanced `n` (A2/D3/D4 are
  n-specific). Unequal `n` is rejected with a structured error (variable-limit
  charts are out of scope for v1).
- **Sigma basis:** within-subgroup, `σ̂ = R̄ / d2` — the R-chart *is* the
  dispersion monitor; the X̄ limits use `A2·R̄ = 3·R̄/(d2·√n) = 3·σ̂/√n`.
- **Source:** [M] §6.3.1, eqs. 6.4–6.6; [AIAG]; [NIST] 6.3.2.

### 3.3 X̄–S chart (subgroups, typically `n >= 9–10`)

`s_j` is the subgroup sample standard deviation (`ddof = 1`).

```
S̄ = (1/k) * sum_j s_j

X̄ chart:  UCL = x-double-bar + A3 * S̄
          CL  = x-double-bar
          LCL = x-double-bar - A3 * S̄

S chart:  UCL = B4 * S̄
          CL  = S̄
          LCL = B3 * S̄        (= 0 for n <= 5, since B3 = 0)
```

- **Constants:** `A3`, `B3`, `B4` from the M1 table at size `n`.
- **Sigma basis:** within-subgroup, `σ̂ = S̄ / c4`; the X̄ limits use
  `A3·S̄ = 3·S̄/(c4·√n) = 3·σ̂/√n`.
- **Assumptions:** as X̄–R; preferred over R for larger `n` because `s` is a more
  efficient dispersion estimator (see §2.2b). Balanced `n` required.
- **Source:** [M] §6.3.2, eqs. 6.24–6.27; [NIST] 6.3.2.

### 3.4 I–MR chart (individuals, `n = 1`)

**Moving-range convention (stated explicitly):** the moving range uses a window
of **w = 2 consecutive observations**, `MR_i = |x_i − x_{i-1}|` for
`i = 2..N`, giving `N − 1` moving ranges. This is the standard default
([M] §6.4; [AIAG]; [NIST] 6.3.2). Larger MR windows exist but are not used; `w =
2` fixes the relevant constants at `d2(2) = 1.128`, `D4(2) = 3.267`,
`D3(2) = 0`, and the **tabulated** `E2(2) = 2.660` (see the convention flag
below — E2 is read from the constants table, not recomputed as `3/d2`).

```
MR̄ = (1/(N-1)) * sum_{i=2..N} |x_i - x_{i-1}|
σ̂  = MR̄ / d2(2) = MR̄ / 1.128

I chart:   UCL = x̄ + E2 * MR̄   ( = x̄ + 3σ̂ )
           CL  = x̄
           LCL = x̄ - E2 * MR̄   ( = x̄ - 3σ̂ )

MR chart:  UCL = D4(2) * MR̄ = 3.267 * MR̄
           CL  = MR̄
           LCL = D3(2) * MR̄ = 0
```

- **`x̄`** is the mean of all `N` individual observations.
- **Assumptions:** observations time-ordered and (ideally) independent; the MR of
  consecutive points is the rational within-variation. **Autocorrelation
  inflates/deflates MR̄** — flagged as a caveat, not silently ignored.
- **Convention flag (resolved in the hardening milestone — B1).** Three E2
  values are defensible and the project previously used a different one in each
  of three places, which is now fixed. The candidates:

  | Candidate | Value | Origin |
  |---|---|---|
  | **Tabulated `E2` (USED)** | **2.660** | published constants table, `CONSTANTS[2].e2` |
  | `3 / d2_table(2)` | 2.65957 | 3 ÷ the rounded tabulated `d2 = 1.128` |
  | `3 / d2_exact(2)` | 2.65868 | 3 ÷ the exact `d2 = 1.1283791…` |

  **The tool uses the tabulated `E2 = 2.660`**, taken directly from the
  validated constants table. Rationale: the within-σ estimator
  `sigma_within_from_individuals` already uses the **tabulated** `d2(2) = 1.128`
  (§2.2), so taking E2 from the same published table keeps the control-chart
  limits and the capability sigma on one convention, and reproduces textbook
  worked examples. `app/spc/limits.py::imr_chart` reads `CONSTANTS[2].e2`
  rather than recomputing `3/d2` inline — there is exactly **one** authoritative
  source, cross-checked by
  `test_e2_matches_authoritative_tabulated_constant`.

  Consequence (disclosed, not hidden): because 2.660 is the rounded published
  value, the I-chart half-width equals `3σ̂` only to ~0.016 %, since
  `3/1.128 = 2.65957`. Earlier versions of this document claimed the exact
  `2.6587` was used; that claim was incorrect and is withdrawn.
- **Source:** [M] §6.4, eqs. 6.28–6.31; [AIAG]; [NIST] 6.3.2.

### 3.5 Zones for the run-rule tests

All run rules operate on **zones** defined by the chart's own CL and its
estimated `σ_points` (the standard deviation of the *plotted* statistic):

- X̄ chart: `σ_points = A2·R̄/3` (X̄–R) or `A3·S̄/3` (X̄–S) — i.e. `(UCL−CL)/3`.
- I chart: `σ_points = E2·MR̄/3 = (UCL−CL)/3`.
- Generic definition used in code: `σ_points = (UCL − CL) / 3`, so zone
  boundaries are exact thirds of the control-limit band. This is the standard
  construction ([M] §5.3; [NIST] 6.3.2).

Zones (each side of CL):
`Zone C = within 1σ`, `Zone B = 1σ..2σ`, `Zone A = 2σ..3σ`, beyond A = outside
limits. Rules are only meaningful on charts with a symmetric ±3σ band (X̄, I);
they are applied to the **X̄ and I charts**. The dispersion charts (R, S, MR)
are one-sided/asymmetric, so only the beyond-limit test (Rule 1) is applied to
them ([NIST] 6.3.2 note).

### 3.6 Western Electric (WECO) rules — definitions

The four classic WECO rules ([Western Electric Statistical Quality Control
Handbook, 1956]; [M] §5.3.1; [NIST] 6.3.2):

| ID | Name | Trigger | Direction |
|---|---|---|---|
| WE1 | Beyond 3σ | 1 point beyond Zone A (outside a control limit) | upper/lower |
| WE2 | 2 of 3 in Zone A or beyond | 2 of 3 consecutive points in Zone A or beyond, same side | upper/lower |
| WE3 | 4 of 5 in Zone B or beyond | 4 of 5 consecutive points in Zone B or beyond, same side | upper/lower |
| WE4 | 8 on one side | 8 consecutive points on the same side of CL | upper/lower |

### 3.7 Nelson rules — definitions

The eight Nelson rules ([Nelson, L.S. (1984), *J. Quality Technology*
16(4):238–239]; [M] §5.3.1 Table 5.1):

| ID | Name | Trigger | Direction |
|---|---|---|---|
| N1 | Beyond 3σ | 1 point beyond a control limit | upper/lower |
| N2 | 9 on one side | 9 consecutive points same side of CL | upper/lower |
| N3 | 6 trending | 6 consecutive points steadily increasing or decreasing | up/down |
| N4 | 14 alternating | 14 consecutive points alternating up and down | none |
| N5 | 2 of 3 in Zone A | 2 of 3 consecutive in Zone A or beyond, same side | upper/lower |
| N6 | 4 of 5 in Zone B | 4 of 5 consecutive in Zone B or beyond, same side | upper/lower |
| N7 | 15 in Zone C | 15 consecutive points within 1σ of CL (both sides) | none |
| N8 | 8 outside Zone C | 8 consecutive points outside 1σ (none in Zone C) **occurring on both sides of the CL** — a mixture signal | none |

**N8 both-sides requirement (implementation note).** Nelson (1984) rule 8 reads
"Eight points in a row on both sides of the centerline with none in Zone C."
Both conditions are constitutive and both are enforced:

1. every point in the run satisfies `|z| >= 1` (outside Zone C), **and**
2. the run contains at least one point above **and** at least one below the CL.

Condition (2) is what makes N8 a **mixture** signal (two overlapping streams)
and keeps it distinct from a one-sided **shift**, which is the job of N2
(9 same side) and WE4 (8 same side). A purely one-sided run beyond 1σ therefore
does **not** trigger N8. Detection scans each maximal outside-Zone-C run for
8-point windows spanning both sides, then merges contiguous/overlapping
qualifying windows so one physical pattern yields one violation.

### 3.8 Differences between the two rule sets

The two sets **overlap but are not identical**, and the tool keeps them
**distinct** (separate rule IDs, separately selectable) rather than merging:

- **WE4 vs N2:** the "one side of CL" run length differs — **WECO uses 8**,
  **Nelson uses 9**. This is the most common point of confusion; both are
  provided under their own IDs.
- **Shared logic, different IDs:** WE1≡N1 (beyond 3σ), WE2≡N5 (2-of-3 Zone A),
  WE3≡N6 (4-of-5 Zone B) share the same test. Under a single-family selection
  they are reported under that family's ID; under `ALL` they are merged into one
  violation carrying both IDs (see the de-duplication bullet below).
- **Nelson-only:** N3 (trend), N4 (alternating), N7 (stratification), N8
  (mixture) have no classic WECO equivalent.
- **Selectable sets:** `WECO` (WE1–WE4), `NELSON` (N1–N8), or `ALL`.
- **`ALL` de-duplication (implemented in the hardening milestone — B2).** When
  both families run, the three logically identical pairs — **WE1↔N1**,
  **WE2↔N5**, **WE3↔N6** — would otherwise each report the same physical event
  twice, inflating `n_violations` and the violations tables in every export.
  They are therefore merged when **all three** match: equivalent rule pair,
  identical affected indices, and identical direction. The merged record keeps
  the Western Electric ID as primary and carries the Nelson ID in
  `Violation.equivalent_rule_ids`, so:
  - `len(violations)` counts **physical events**, not detectors;
  - both IDs remain attributable — `Violation.all_rule_ids`,
    `Violation.matches_rule("N1")` and `RuleEvaluation.triggered_rule_ids` all
    still surface the Nelson ID;
  - **WE4 (8 on one side) and N2 (9 on one side) are NOT merged** — the run
    lengths differ, so they are genuinely different claims about the data;
  - Nelson-only rules (N3, N4, N7, N8) pass through untouched.

  Default = `NELSON` (superset of practical value).

> **Convention flag (run lengths).** Where sources disagree (WE 8-in-a-row vs
> Nelson 9-in-a-row; some texts use 7 or 8 for trends), the tool uses the
> **canonical published length for each named rule** and documents it here; it
> does not silently pick one length and apply it to both rule sets.

### 3.8a Physical events vs sliding-window detections (C1)

The k-of-m rules (WE2/N5, WE3/N6) are evaluated with a sliding window, but a
**window is the detector, not the event**. A sustained excursion satisfies the
test at several consecutive window positions; emitting one violation per window
would make a single special cause look like several independent ones and would
inflate `n_violations` everywhere it is quoted.

Overlapping qualifying windows **on the same side** are therefore merged into a
single violation whose `indices` are the union of the flagged points. Merging is
by window adjacency, **not** by `set(indices)`, so genuinely distinct events stay
distinct:

- a non-qualifying window position (a gap) **closes** the event — two excursions
  separated by in-control points are reported as two violations;
- the upper and lower sides are tracked **separately** — an excursion above the
  CL followed by one below yields two violations with the correct directions;
- points that do not themselves clear the zone are never flagged, even when they
  sit inside a qualifying window (4-of-5 permits one interior miss).

### 3.8b Flat-line (zero-variation) charts and N7 (B4)

When a series is exactly constant, the chart has `MR̄ = 0`, hence
`UCL = LCL = CL` and `sigma_points = 0`. The signed zone level is defined as
`z = 0` for every point (avoiding 0/0), with a deliberate consequence:

**N7 (15 consecutive points within 1σ) DOES fire on a flat line of ≥ 15
points, and it is the only rule that fires.**

This is the intended behavior, not an oversight. N7 detects **stratification** —
points hugging the center line far more tightly than the control limits imply —
and an exactly flat series is the limiting case. In manufacturing this almost
always means the **measurement resolution is too coarse** for the process, or
the data were rounded/duplicated; both are real, actionable findings ([M] §5.3
on limits too wide relative to the plotted statistic's true dispersion). An
earlier docstring claimed "no run rule can fire on a flat line"; that claim was
false and has been corrected rather than the behavior being suppressed.

Rules that **cannot** fire on a flat line, for reference: the beyond-limit test
(strict inequality, and `UCL == LCL == CL`); same-side runs WE4/N2 (points on
the CL belong to neither side); N3 (ties break trends); N4 (ties break
alternation); WE2/WE3/N5/N6 (`|z| = 0` clears no zone); N8 (`|z| = 0` is inside
Zone C).

To avoid contradictory advice, the interpretation layer's `zero_variation`
conclusion explicitly frames this N7 signal as stratification / inadequate gauge
resolution **rather than** an ordinary process shift, and the
`rec_maintain` recommendation is suppressed entirely for zero-variation data
(§5.5).

### 3.9 Missing / insufficient data handling

- **Missing/non-finite values:** control-chart inputs are validated up front. A
  subgroup or individual series containing NaN/inf is **not silently dropped**;
  the calculator records the positions and either (a) for individuals, excludes
  them from MR computation while **preserving original indices** in the result so
  violation indices still map to the user's rows, or (b) raises a structured
  error if a whole subgroup is unusable — the caller decides. The default policy
  is documented per function.
- **Insufficient data:** each rule needs a minimum run length (e.g. N2 needs
  ≥9 points). When the series is shorter than a rule's window, that rule simply
  cannot trigger — it is skipped, and a structured `RuleSkipped` note records
  which rules were not evaluated and why. Too few subgroups to form limits
  (`k < 2`) raises a structured error. None of this crashes.
- **Precision:** all limits and statistics are computed at full float precision;
  no rounding before comparison (a point exactly on a limit is treated as **not**
  beyond it — limits are inclusive of the boundary, `strictly greater`/`strictly
  less` triggers a violation; documented per rule).

---

## 4. Chart construction and rendering (presentation only)

**The chart layer introduces no new statistical formulas.** It is a presentation
layer that consumes the validated `ControlChart` and `RuleEvaluation` objects (§3) and turns
them into visual charts. The architectural invariant is enforced in code and by
tests:

- **Single source of truth.** CL, UCL, LCL, plotted point values, original
  observation indices, and rule violations come *only* from the §3 calculator
  output. The chart layer never recomputes a limit or re-evaluates a rule. A
  dedicated test asserts the plotted/annotated values are identical (bit-for-bit
  via `==`) to the source objects.
- **Separation of concerns.** `charts.model` builds a pure, plot-library-agnostic
  `ChartModel` from the §3 objects (no matplotlib import); `charts.render`
  renders a `ChartModel` to a matplotlib figure and exports PNG/SVG. Data
  cleaning (§1), statistics (§2), and rule evaluation (§3) are untouched.
- **Point classification for display** (derived, not recomputed): each plotted
  point is tagged `NORMAL`, `BEYOND_LIMITS` (from the chart's `beyond_limits`
  flag), and/or `RULE_VIOLATION` (its original index appears in some `Violation`).
  These are display categories layered on top of the §3 truth, never a
  re-derivation of it.
- **Accessibility:** out-of-control state is shown by **marker shape and outline
  in addition to color** (not color alone), per the requirement.
- **Units:** the measurement unit (when present) is placed in the Y-axis label;
  the X-axis label is "Observation" (I-MR) or "Subgroup" (X-bar/R, X-bar/S)
  according to chart type.

This layer contributes no statistical content to this reference.

---

## 5. Interpretation engine and report serialization

**The interpretation and export layers introduce no new statistical formulas.**
They consume the validated §1-§3 quantities only. This section documents the
*interpretation thresholds* (engineering conventions, not statistical laws) and
the rule triggers, so every automated conclusion is auditable.

### 5.1 Statistical conclusions vs engineering recommendations

The engine emits two strictly separated outputs (separate classes, separate
lists): **statistical conclusions** are factual statements about the numbers;
**engineering recommendations** are suggested actions tied to a condition and are
never presented as statistical fact. Stability has priority: an unstable process
yields a CRITICAL conclusion that capability must not be read as reliable, while
the numeric indices are always retained (never nulled).

### 5.2 Capability category bands (engineering convention — NOT a statistical law)

Applied to Cpk/Ppk (and Cp/Pp where relevant). Defaults follow common AIAG-style
practice and are **configurable** via `InterpretationThresholds`:

| Band | Condition (default) | Meaning |
|---|---|---|
| INADEQUATE | index < 1.00 | inadequate margin |
| MARGINAL | 1.00 <= index < 1.33 | barely adequate |
| CAPABLE | 1.33 <= index < 1.67 | adequate margin |
| EXCELLENT | index >= 1.67 | substantial margin |

> **Convention flag.** These cut-points are an interpretation convention that
> varies by industry and customer standard (e.g. automotive PPAP often requires
> 1.67 for new processes, 1.33 ongoing). They are exposed as configuration and
> are explicitly labelled as conventions in the generated conclusion text; they
> are **not** universal statistical thresholds.

### 5.3 Other interpretation thresholds (documented, configurable)

| Threshold | Default | Rule it drives |
|---|---|---|
| `centering_gap_ratio` | 0.10 | Cpk is "materially below" Cp when `(Cp−Cpk)/Cp >= 0.10` -> off-centre conclusion + centering recommendation. |
| `drift_gap_ratio` | 0.10 | Ppk is "materially worse" than Cpk when `(Cpk−Ppk)/Cpk >= 0.10` -> long-term-drift conclusion + drift recommendation. |
| `spec_margin_tight_sigma` | 6.0 | Two-sided tolerance flagged tight when width `< 6.0 * sigma_within` (narrower than the natural +/-3σ band). |
| `min_observations` | 30 | Small-sample conclusion + collect-more-data recommendation when N < 30. |
| `min_subgroups` | 20 | Few-subgroups notice when subgroup count < 20. |

These mirror the §2.7 reliability-guard thresholds so the interpreter's language is
consistent with the guards attached at analysis time.

### 5.4 Rule triggers (deterministic; each conclusion has a stable code)

| Trigger code | Condition | Emits |
|---|---|---|
| `process_unstable` | stability assessed & not stable | CRITICAL conclusion + investigate-stability rec |
| `process_stable` | stability assessed & stable | INFO conclusion; maintain rec only if the §5.4a evidence gates all pass |
| `stability_not_assessed` | no stability supplied | NOTICE conclusion |
| `zero_variation` | sigma_within or sigma_overall == 0 | WARNING conclusion + verify-gauge rec |
| `one_sided_spec` | spec has one bound | INFO conclusion (no false Cp/Pp talk) |
| `capability_*` / `performance_*` | Cpk/Ppk band | band conclusion (severity by band) |
| `off_center` | two-sided, `(Cp−Cpk)/Cp >= centering_gap_ratio` | WARNING conclusion + centering rec |
| `well_centered` | two-sided, gap below ratio | INFO conclusion |
| `excessive_variation` | two-sided, Cp<1 and Cpk<1 | WARNING conclusion + reduce-variation + review-spec recs |
| `long_term_drift` | `(Cpk−Ppk)/Cpk >= drift_gap_ratio` | WARNING conclusion + investigate-drift rec |
| `within_overall_consistent` | drift gap below ratio | INFO conclusion |
| `spec_margin_tight` | two-sided width `< 6σ_within` | NOTICE conclusion |
| `small_sample` | N < min_observations | WARNING conclusion + collect-more-data rec |
| `few_subgroups` | subgroups < min_subgroups | NOTICE conclusion |
| `non_normal` | normality rejected | WARNING conclusion + check-distribution rec |
| `cpk_confidence_interval` | CI present | INFO conclusion (states it is approximate) |
| `rule_violations_detail` | unstable with rule IDs | WARNING conclusion (lists rules) |

A rule fires **only** when the data it needs is present (e.g. centering logic
requires a two-sided spec so Cp exists) — the engine never invents a conclusion
from missing data.

#### 5.4a Evidence gating for the MAINTAIN recommendation (B3)

`rec_maintain` ("the process is stable and capable — maintain current controls")
is the strongest reassurance the tool emits, so it requires positive evidence on
**every** axis. It is emitted only when all of the following hold:

1. stability was **assessed** and the process is **stable**;
2. a capability report exists and `Cpk` is **finite** and in the CAPABLE or
   EXCELLENT band;
3. normality was **not rejected**;
4. the sample size meets `min_observations`.

Conditions 2–4 were added in the hardening milestone. Previously a
zero-variation dataset produced `Cpk = +inf` → EXCELLENT → a maintain
recommendation emitted **alongside** the contradictory "zero variation is almost
always a gauge artifact — verify before trusting any index" warning. A `+inf`
index is a measurement-resolution artifact, not capability evidence; likewise a
flattering Cpk from 12 observations or from data that fail a normality test does
not support "carry on as you are". When a gate fails the tool stays **silent**
on maintenance and lets the corresponding warning carry the message.

This gating applies to the **engineering recommendation only**. The statistical
conclusions (including the index-band conclusions and the `zero_variation`
conclusion) are unaffected, preserving the strict separation between what the
statistics *say* and what the engineer is *advised to do*.

### 5.5 Export layers (no statistics)

CSV and Excel exporters serialize the analysis + interpretation. They perform **no
statistical computation**: all values come from the tested Python analysis layer.
Missing/unavailable values are represented explicitly (e.g. `"n/a"`), never
silently dropped; full numeric precision is preserved in CSV. Excel uses separate
logical sheets (Summary, Measurements, Statistics, Capability, SPC, Violations,
Interpretation).

---

## Document status

This reference is complete for the released version. The PDF report's Technical
Appendix cites this document as the controlled methodology source; the PDF and
the UI never restate formulas independently. Any future statistical addition
(e.g. non-normal capability methods, variable-limit charts) must be documented
here **before** implementation, following the same formula / assumptions /
convention-flag / reference format.
