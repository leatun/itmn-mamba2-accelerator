# Shared by the run_*_sim.do scripts (run from this folder). Optional Tcl variables:
#   M2_SV, M2_PSHV  d_state / 16 and log2(headdim / 16) (default 1, 1; needs the matching layout and goldens)
#   GOLD_DIR        golden root, forward slashes (default ../goldens/itmn_m2)
#   LIBN            work library (default given by the script), so several runs can share the folder
if {[info exists M2_SV]}    { append defs " +define+M2_S=$M2_SV +define+M2_S$M2_SV" }
if {[info exists M2_PSHV]}  { append defs " +define+M2_PSH=$M2_PSHV" }
if {[info exists GOLD_DIR]} { append defs " +define+GOLDEN_ROOT=\\\"$GOLD_DIR\\\"" }
if {![info exists LIBN]}    { set LIBN $DEFAULT_LIB }

set RTL "src/common/BRAM_256b.v src/common/Exp_LUT.v src/common/Silu_LUT.v src/common/Softplus_LUT.v \
         src/common/RSqrt_ROM.v src/common/RSqrt_FP_ROM.v src/common/LUT_Bank.v src/common/Memory_Cluster.v \
         src/mamba/H_RegFile.v src/mamba/Reduce16Wide.v src/mamba/Mamba_PE_v2.v src/mamba/M_Cluster.v \
         src/mamba/Mamba_Cluster.v src/top/Mamba_Top.v"

proc run_tb {tb} {
    global defs RTL LIBN
    vlib $LIBN
    eval vlog -work $LIBN +incdir+src/common +incdir+tb -O0 $defs $RTL tb/$tb.v
    vsim -c -lib $LIBN -voptargs=+acc $tb
    run -all
    quit -f
}
