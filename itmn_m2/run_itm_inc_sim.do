# Inception phase of one block. Tcl vars: ITM_BK (block, default 0), INC_T (first N timesteps).
# Common options in sim_common.tcl.
set defs ""
if {[info exists ITM_BK]} { append defs " +define+ITM_BK=$ITM_BK" }
if {[info exists INC_T]} { append defs " +define+INC_T=$INC_T" }
set DEFAULT_LIB work_inc
source sim_common.tcl
run_tb tb_ITM_INC
