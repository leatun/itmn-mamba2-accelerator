# Mamba2 mixer of one block. Tcl vars: ITM_BK (block, default 0), MIX_T (timesteps, default 4).
# Common options in sim_common.tcl.
set defs ""
if {[info exists ITM_BK]} { append defs " +define+ITM_BK=$ITM_BK" }
if {[info exists MIX_T]} { append defs " +define+MIX_T=$MIX_T" }
set DEFAULT_LIB work_m2mix
source sim_common.tcl
run_tb tb_M2_MIX
