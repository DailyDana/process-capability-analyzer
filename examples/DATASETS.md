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
