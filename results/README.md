# Results

Kria KV260 (xck26-sfvc784-2LV-c), 250 MHz (4.0 ns), Vivado 2022.1, out-of-context place and route.

## End to end: one ECG (12 leads x 1000 samples) to 5 logits

From `tb_ITM_E2E` (full length; every block output, the GAP and the logits checked against the goldens):

| Block | T | d_model | Inception phase | Mamba2 phase (+ block output, pooling, GAP and FC on block 4) | Total |
|---|---|---|---|---|---|
| 0 (with encoder) | 1000 | 64 | 229,566 | 452,191 | 681,757 |
| 1 | 1000 | 64 | 207,557 | 452,191 | 659,748 |
| 2 | 500 | 64 | 103,807 | 226,191 | 329,998 |
| 3 | 500 | 64 | 103,807 | 226,191 | 329,998 |
| 4 (with GAP, FC) | 250 | 128 | 175,807 | 296,953 | 472,760 |
| **all** | | | **820,544** | **1,653,717** | **2,474,261 = 9.90 ms** |

Logits of record 0: 3488, -2087, -3410, -1794, -3469. Weight loading between phases is not included.

## Mamba2 mixer: cycles per timestep by stage

From `tb_M2_MIX` at T = 8, steady state, one cluster's view (`cycle_profile.txt` has all configurations).

| Stage | What it does | Block 0 (d_model 64) | Block 4 (d_model 128) |
|---|---|---|---|
| RN | block-input RMSNorm (apply only; the sum of squares runs on the gate row during the previous timestep) | 5 | 6 |
| M1 | in_proj (z, x, B, C, dt), split-K over the PE array and the scan rows | 196 | 636 |
| M2 | causal conv (width 4) with bias and SiLU | 32 | 48 |
| M6 | scan, with dt, exp(dt A), D x and the SiLU(z) gate fused in | 101 | 173 |
| GN | gated RMSNorm (sums of squares already accumulated on the gate row) | 28 | 36 |
| M8 | out_proj, split-K | 76 | 274 |
| barrier | wait for both clusters and every mirrored write | 15 | 12 |
| **per timestep** | measured over the run, including the one-off state clear | **470** | **1,217** |

The scan at larger state sizes (d_state 32 / 64 / 128, block 0): 165 / 294 / 558 cycles per timestep, all checked
against goldens of that size during development (goldens not included here).

## Post-route resources and timing

`route_kv260/` (from `itmn_m2/run_synth_kv260.tcl`):

| WNS (setup) | WHS (hold) | LUT | FF | BRAM tiles | URAM | DSP | Power (vectorless) |
|---|---|---|---|---|---|---|---|
| +0.071 ns | +0.052 ns | 56,072 (48 %) | 37,878 (16 %) | 116 / 144 | 32 / 64 | 146 / 1,248 | 2.03 W |

By module (`route_kv260/util_hier_route.rpt`), per cluster: PE array 16 lanes x 2 DSP = 32 DSP and about 8.6-9.0 k LUT;
controller with the scan rows and the MAC datapath about 16.5 k LUT and 41 DSP; Reduce16Wide 0.4 k LUT. The two
clusters together are 146 DSP.
