# One ITM block, both phases. Tcl vars: ITM_BK (block, default 0), BLK_T (first N timesteps).
# Common options in sim_common.tcl.
set defs ""
if {[info exists ITM_BK]} { append defs " +define+ITM_BK=$ITM_BK" }
if {[info exists BLK_T]} { append defs " +define+BLK_T=$BLK_T" }
set DEFAULT_LIB work_blk
source sim_common.tcl
run_tb tb_ITM_BLOCK
