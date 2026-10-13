# Example datasets

## pistonrings.csv / pistonrings_phase1.csv — real production data

Inside diameter (mm) of piston rings for an automotive engine forging process.
40 samples of 5 rings, in production order (200 measurements).

* **Source:** dataset `pistonrings` from the R package `qcc` (Luca Scrucca).
  The measurements originate from Montgomery, *Introduction to Statistical
  Quality Control*, Tables 6.3/6.4 — the canonical worked X-bar/R example.
* **Specification:** nominal 74.000 mm, +/-0.05 -> LSL 73.95, USL 74.05.
* **Structure:** samples 1-25 are Phase I (limit establishment); samples 26-40
  are Phase II (monitoring), where the process drifts upward.
* `pistonrings_phase1.csv` is samples 1-25 only (125 measurements).

Because this dataset is published with worked results, it doubles as an
end-to-end check on the implementation. Published values vs. this application:

| quantity     | published | this application (Phase I) |
|--------------|-----------|----------------------------|
| x-bar-bar    | 74.001    | 74.00118                   |
| R-bar        | 0.023     | 0.02276                    |
| UCL (X-bar)  | 74.014    | 74.01431                   |
| LCL (X-bar)  | 73.988    | 73.98804                   |

Phase I is reported STABLE with 0 rule violations, Cp 1.70 / Cpk 1.66. All 40
samples together are reported UNSTABLE (N1, N5, N6): the Phase II drift is
caught.

Suggested settings: X-bar/R chart, subgroup size 5, Nelson rules.

## sample_measurements.csv — synthetic

58 observations: a stable process seeded with one gross outlier and a sustained
shift at the end. Built to exercise the rule engine end to end; it is not real
measurement data. Analyzed as I-MR with USL 11.5 / LSL 8.5.

## gage_rr_aiag.csv — AIAG MSA reference Gage R&R study

Crossed Gage R&R study: 10 parts, 3 appraisers (A, B, C), 3 trials, 90
readings. Long format: `part, operator, trial, value`.

* **Source:** the example study of the AIAG *Measurement Systems Analysis
  Reference Manual*, 4th ed. (2010). Readings transcribed from BPI Consulting,
  ["Three Ways to Analyze a Gage R&R Study"](https://www.spcforexcel.com/Downloads/pdf/Three-Ways-to-Analyze-a-Gage-RR.pdf)
  (2015), Table 1. The same data is the subject of Minitab's
  ["Example of Crossed Gage R&R Study"](https://support.minitab.com/minitab/help-and-how-to/quality-and-process-improvement/measurement-system-analysis/how-to/gage-study/crossed-gage-r-r-study/before-you-start/example).
  The transcription is confirmed by the sums of squares, which match Minitab's
  published ANOVA to every printed decimal.
* **Tolerance:** 8 (USL − LSL), as in Minitab's analysis.

Published results vs. this application (ANOVA, k = 6, tolerance 8; the
interaction p = 0.974 is pooled):

| source          | published %Study Var | app   | published %Tolerance | app   |
|-----------------|---------------------:|------:|---------------------:|------:|
| Total Gage R&R  | 27.86                | 27.86 | 22.68                | 22.68 |
| Repeatability   | 18.42                | 18.42 | 14.99                | 14.99 |
| Reproducibility | 20.90                | 20.90 | 17.01                | 17.01 |
| Part-to-Part    | 96.04                | 96.04 | 78.17                | 78.17 |

ndc = 4 in both. With the Average & Range method the application gives
EV 0.202, AV 0.230, GRR 0.306, PV 1.105, TV 1.146 and %GRR 26.68, which matches
BPI's Table 2. The full set of compared figures is in
`tests/test_msa_published.py`.

The range chart flags one cell: appraiser B on part 4 reads 0.01, 1.03, 0.20
(R = 1.02 > UCL 0.880). This is the application's own finding, not a published
figure.

## gage_rr_thermal_impedance.csv — Houf & Burman (1988)

Thermal impedance (°C/W × 100) of power modules: 10 parts, 3 operators, 3
trials.

* **Source:** Houf & Burman (1988), as listed and analysed in SAS/STAT 14.1,
  [PROC VARCOMP Example 121.2](https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_varcomp_examples02.htm).
* **Specification:** 18 to 58 (tolerance 40), as in the SAS example.

Unlike the AIAG data, the operator × part interaction is strongly significant
(p ≈ 5e-7) and stays in the model. Published SAS results vs. this application:

| quantity           | SAS        | app        |
|--------------------|-----------:|-----------:|
| Var(part)          | 48.29259   | 48.29259   |
| Var(operator)      | 0.56461    | 0.56461    |
| Var(interaction)   | 0.72798    | 0.72798    |
| Var(error)         | 0.51111    | 0.51111    |
| GRR variance       | 1.80370    | 1.80370    |
| PTR (18, 58, k=6)  | 0.20145    | 0.20145    |

The two studies together cover both branches of the ANOVA (interaction pooled,
interaction kept). The readings are reproduced here, with attribution, only to
verify this implementation.
