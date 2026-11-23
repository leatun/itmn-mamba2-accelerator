# itmn_m2: ITMN-Mamba2 accelerator

Verilog accelerator for the ITMN ECG classifier (encoder, 5 ITM blocks of Inception and a Mamba2 mixer, GAP, FC)
in Q4.11 fixed point, two clusters of 16 lanes, KV260 at 250 MHz. The testbenches compare against the goldens in
`../goldens/itmn_m2` (one test record, d_state 16, headdim 32).

## Layout

| Path | Contents |
|---|---|
| `src/top/Mamba_Top.v` | two clusters, their memories, mirror writes between them, host DMA |
| `src/mamba/Mamba_Cluster.v` | controller of one cluster (all stages), scan fabric rows |
| `src/mamba/M_Cluster.v`, `Mamba_PE_v2.v` | 16-lane PE array (MAC, MAC2, MUL, MUL2) and the scan state RAM |
| `src/mamba/Reduce16Wide.v`, `H_RegFile.v` | lane-reduce tree, state RAM |
| `src/common/` | parameters, memory layout, RAM wrapper, activation LUTs, rsqrt ROMs |
| `tb/` | testbenches (below) |
| `scripts/m2_layout.py` | generates and checks `src/common/m2_layout.vh` |

## Simulation (ModelSim, run from this folder)

| Script | Testbench | Checks |
|---|---|---|
| `run_m2_mix_sim.do` | `tb_M2_MIX` | Mamba2 mixer of one block, every intermediate |
| `run_itm_inc_sim.do` | `tb_ITM_INC` | Inception phase of one block |
| `run_itm_block_sim.do` | `tb_ITM_BLOCK` | one ITM block, both phases (+ GAP and FC on block 4) |
| `run_itm_e2e_sim.do` | `tb_ITM_E2E` | raw 12-lead ECG to logits, all blocks |

    vsim -c -do "set ITM_BK 4; set MIX_T 8; do run_m2_mix_sim.do"
    vsim -c -do "set ITM_BK 1; set BLK_T 60; do run_itm_block_sim.do"
    vsim -c -do "do run_itm_e2e_sim.do"

Shared options (`sim_common.tcl`): `M2_SV` and `M2_PSHV` select d_state = 16 * M2_SV and headdim = 16 << M2_PSHV
(with goldens of that size in `GOLD_DIR`; only the default size is included), `LIBN` the work library.

## Interface rules

- Host DMA writes go to the main RAM only in a cycle with no core or mirror write, and DMA reads take the core's
  read port; there is no arbitration or stall back to the host. Use the DMA (loads and read-back) only while the
  accelerator is idle (between `done_all` and `start`).
- The bank X weights (split-K share) are stored transposed; the host must write them in that layout (see
  `tb/tb_wsplit.vh`, `scripts/splitk_model.py`).
- Supported geometry (asserted at start in simulation): d_inner = 2 d_model, d_inner 128 or 256, CH_OUT 4 or 8,
  at most 16 heads, T <= 512 when the GAP accumulates.
- Cycle counts in the testbenches are compute only, from `start` to `done_all` per phase; weight reloads between
  phases are outside them.

## Synthesis (Vivado, out of context)

The route at 4.0 ns is out of context: ideal clock (no clock tree), all inputs false-pathed, vectorless power.

    vivado -mode batch -source run_synth_kv260.tcl -tclargs 4.0
