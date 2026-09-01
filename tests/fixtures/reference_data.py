"""Independently hand-calculated reference datasets for capability tests.

Every expected value in this module was computed by hand (shown in the
derivation comments), NOT by running the implementation under test. Control
constants used come from the published tables:

    d2(2) = 1.128,  d2(3) = 1.693,  c4(3) = 0.8862,  c4(4) = 0.9213

z_{0.025} = 1.959964 (standard normal 97.5th percentile).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# DRIFT dataset: tiny within-subgroup spread, large between-subgroup shift.
# Proves sigma_within != sigma_overall and Cp/Cpk != Pp/Ppk.
# ---------------------------------------------------------------------------
# Three subgroups of size 3, each of the form [x, x+1, x+2]:
DRIFT_SUBGROUPS = [
    np.array([10.0, 11.0, 12.0]),
    np.array([20.0, 21.0, 22.0]),
    np.array([30.0, 31.0, 32.0]),
]
DRIFT_ALL = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0])
DRIFT_SPEC_USL = 40.0
DRIFT_SPEC_LSL = 0.0

# --- within sigma via R-bar/d2 ---
# Each subgroup range R = 2. R-bar = 2. d2(3) = 1.693.
# sigma_within = 2 / 1.693 = 1.181334...
DRIFT_RBAR = 2.0
DRIFT_D2_N3 = 1.693
DRIFT_SIGMA_WITHIN = 2.0 / 1.693  # 1.1813349...

# --- within sigma via S-bar/c4 (forced) ---
# Each subgroup s (ddof=1) = sqrt(((−1)^2+0^2+1^2)/2) = sqrt(1) = 1.0. S-bar=1.0.
# c4(3) = 0.8862.  sigma = 1.0 / 0.8862 = 1.128413...
DRIFT_SBAR = 1.0
DRIFT_C4_N3 = 0.8862
DRIFT_SIGMA_WITHIN_SBAR = 1.0 / 0.8862  # 1.1284134...

# --- overall sigma ---
# Grand mean: sum = 33+63+93 = 189; mean = 189/9 = 21.0
# Squared deviations from 21: 121,100,81,1,0,1,81,100,121 -> sum = 606
# sample variance = 606 / 8 = 75.75 ; sigma_overall = sqrt(75.75) = 8.703447...
DRIFT_MEAN = 21.0
DRIFT_SS = 606.0
DRIFT_VAR_OVERALL = 606.0 / 8.0  # 75.75
DRIFT_SIGMA_OVERALL = np.sqrt(75.75)  # 8.7034475...

# --- capability (within) with USL=40, LSL=0, mean=21 ---
# Cp  = (40-0)/(6*1.181334) = 40 / 7.088006 = 5.643356
# Cpu = (40-21)/(3*1.181334) = 19 / 3.544003 = 5.361456
# Cpl = (21-0)/(3*1.181334) = 21 / 3.544003 = 5.925821
# Cpk = min = 5.361456
DRIFT_CP = 40.0 / (6.0 * DRIFT_SIGMA_WITHIN)
DRIFT_CPU = 19.0 / (3.0 * DRIFT_SIGMA_WITHIN)
DRIFT_CPL = 21.0 / (3.0 * DRIFT_SIGMA_WITHIN)
DRIFT_CPK = DRIFT_CPU  # the smaller of Cpu/Cpl

# --- performance (overall) ---
# Pp  = 40/(6*8.703447) = 40/52.22068 = 0.765986
# Ppu = 19/(3*8.703447) = 19/26.11034 = 0.727681
# Ppl = 21/(3*8.703447) = 21/26.11034 = 0.804282
# Ppk = min = 0.727681
DRIFT_PP = 40.0 / (6.0 * DRIFT_SIGMA_OVERALL)
DRIFT_PPU = 19.0 / (3.0 * DRIFT_SIGMA_OVERALL)
DRIFT_PPL = 21.0 / (3.0 * DRIFT_SIGMA_OVERALL)
DRIFT_PPK = DRIFT_PPU


# ---------------------------------------------------------------------------
# SYMMETRIC dataset: mean exactly at spec center -> Cp == Cpk.
# ---------------------------------------------------------------------------
# Values [8,9,10,11,12]: mean = 10.
# Deviations -2,-1,0,1,2 -> squares 4,1,0,1,4 = 10 ; var = 10/4 = 2.5
# sigma = sqrt(2.5) = 1.5811388...
SYMM_VALUES = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
SYMM_MEAN = 10.0
SYMM_SIGMA = np.sqrt(2.5)  # 1.5811388

# Centered spec: USL=16, LSL=4, center = 10 = mean.
CENTERED_USL = 16.0
CENTERED_LSL = 4.0
# Cp  = (16-4)/(6*1.5811388) = 12 / 9.4868330 = 1.264911
# Cpu = (16-10)/(3*1.5811388) = 6 / 4.7434165 = 1.264911
# Cpl = (10-4)/(3*1.5811388) = 6 / 4.7434165 = 1.264911
# Cpk = min = 1.264911  ==  Cp   (perfectly centered)
CENTERED_CP = 12.0 / (6.0 * SYMM_SIGMA)
CENTERED_CPK = 6.0 / (3.0 * SYMM_SIGMA)

# Off-center spec: USL=20, LSL=4, center = 12, mean = 10 (off by 2).
OFFCENTER_USL = 20.0
OFFCENTER_LSL = 4.0
# Cp  = (20-4)/(6*1.5811388) = 16 / 9.4868330 = 1.686548
# Cpu = (20-10)/(3*1.5811388) = 10 / 4.7434165 = 2.108185
# Cpl = (10-4)/(3*1.5811388) = 6  / 4.7434165 = 1.264911
# Cpk = min = 1.264911 < Cp = 1.686548
OFFCENTER_CP = 16.0 / (6.0 * SYMM_SIGMA)
OFFCENTER_CPU = 10.0 / (3.0 * SYMM_SIGMA)
OFFCENTER_CPL = 6.0 / (3.0 * SYMM_SIGMA)
OFFCENTER_CPK = OFFCENTER_CPL


# ---------------------------------------------------------------------------
# ONE-SIDED specs on the symmetric dataset (mean=10, sigma=1.5811388).
# ---------------------------------------------------------------------------
# Upper only USL=16:  Cp=None, Cpu=Cpk=(16-10)/(3*sigma)=1.264911, Cpl=None
ONESIDED_UPPER_USL = 16.0
ONESIDED_UPPER_CPK = 6.0 / (3.0 * SYMM_SIGMA)
# Lower only LSL=4:   Cp=None, Cpl=Cpk=(10-4)/(3*sigma)=1.264911, Cpu=None
ONESIDED_LOWER_LSL = 4.0
ONESIDED_LOWER_CPK = 6.0 / (3.0 * SYMM_SIGMA)


# ---------------------------------------------------------------------------
# ZERO-VARIATION dataset.
# ---------------------------------------------------------------------------
ZERO_VALUES = np.array([10.0, 10.0, 10.0, 10.0])
ZERO_USL = 16.0
ZERO_LSL = 4.0
# sigma = 0 -> Cp = Cpk = +inf (mean 10 strictly inside spec)


# ---------------------------------------------------------------------------
# SIGMA-LEVEL reference: round Z on both sides.
# mean=10, sigma=2, USL=16, LSL=4  ->  Z_USL = Z_LSL = 3 exactly.
# ---------------------------------------------------------------------------
SIGLEVEL_MEAN = 10.0
SIGLEVEL_SIGMA = 2.0
SIGLEVEL_USL = 16.0
SIGLEVEL_LSL = 4.0
# Phi(-3) = 0.00134989803 (standard tabulated value)
SIGLEVEL_P_ABOVE = 0.00134989803
SIGLEVEL_P_BELOW = 0.00134989803
SIGLEVEL_P_TOTAL = 2.0 * 0.00134989803  # 0.0026997960
SIGLEVEL_DPMO = SIGLEVEL_P_TOTAL * 1e6  # 2699.796
# Z_bench = Phi^{-1}(1 - 0.0026997960) = Phi^{-1}(0.997300204) = 2.78215 (hand)
SIGLEVEL_ZBENCH = 2.78215


# ---------------------------------------------------------------------------
# CPK CONFIDENCE-INTERVAL references (Bissell 1990 / Montgomery 8.24).
# CI = cpk +/- z * sqrt( 1/(9N) + cpk^2/(2(N-1)) ),  z_{0.025} = 1.959964
# ---------------------------------------------------------------------------
Z_025 = 1.959964

# Case A: cpk=1.0, N=50
# se = sqrt(1/450 + 1/98) = sqrt(0.00222222 + 0.01020408) = sqrt(0.01242630)
#    = 0.11147333
# margin = 1.959964 * 0.11147333 = 0.21848076
# lower = 0.78151924 ; upper = 1.21848076
CI_A_CPK = 1.0
CI_A_N = 50
CI_A_SE = np.sqrt(1.0 / 450.0 + 1.0 / 98.0)
CI_A_LOWER = 1.0 - Z_025 * CI_A_SE
CI_A_UPPER = 1.0 + Z_025 * CI_A_SE

# Case B: cpk=1.33, N=100
# se = sqrt(1/900 + 1.7689/198) = sqrt(0.00111111 + 0.00893384) = sqrt(0.01004495)
#    = 0.10022450
# margin = 1.959964 * 0.10022450 = 0.19643573
# lower = 1.13356427 ; upper = 1.52643573
CI_B_CPK = 1.33
CI_B_N = 100
CI_B_SE = np.sqrt(1.0 / 900.0 + (1.33**2) / (2.0 * 99.0))
CI_B_LOWER = 1.33 - Z_025 * CI_B_SE
CI_B_UPPER = 1.33 + Z_025 * CI_B_SE


# ---------------------------------------------------------------------------
# WITHIN-ESTIMATOR references.
# ---------------------------------------------------------------------------
# Pooled (unbalanced): subgroup A=[10,11,12] (n=3, s=1, dof2),
#                      subgroup B=[20,22]    (n=2, s=sqrt(2), dof1)
# pooled ss = 2*1^2 + 1*(sqrt(2))^2 = 2 + 2 = 4 ; dof_total = 3
# s_pooled = sqrt(4/3) = 1.1547005 ; c4(4) = 0.9213
# sigma = 1.1547005 / 0.9213 = 1.253339
POOLED_SUBGROUPS = [np.array([10.0, 11.0, 12.0]), np.array([20.0, 22.0])]
POOLED_S = np.sqrt(4.0 / 3.0)
POOLED_C4_N4 = 0.9213
POOLED_SIGMA = np.sqrt(4.0 / 3.0) / 0.9213

# Moving range (individuals): [10,12,11,14] -> MR = 2,1,3 ; MR-bar = 2.0
# d2(2) = 1.128 ; sigma = 2.0/1.128 = 1.773050
MR_VALUES = np.array([10.0, 12.0, 11.0, 14.0])
MR_BAR = 2.0
MR_D2 = 1.128
MR_SIGMA = 2.0 / 1.128
