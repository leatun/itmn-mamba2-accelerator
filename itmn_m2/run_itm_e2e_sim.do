# End to end, raw ECG -> logits. Tcl var: ITM_BK (stop after this block, default 4).
# Common options in sim_common.tcl.
set defs ""
if {[info exists ITM_BK]} { append defs " +define+E2E_LAST=$ITM_BK" }
set DEFAULT_LIB work_e2e
source sim_common.tcl
run_tb tb_ITM_E2E
