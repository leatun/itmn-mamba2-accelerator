# Out-of-context synthesis and implementation of Mamba_Top for the KV260 (xck26-sfvc784-2LV-c). Out of context
# because the 256-bit DMA buses exceed the package I/O; the host interface is not part of the measured design.
#
#   vivado -mode batch -source run_synth_kv260.tcl -tclargs [period_ns] [report_dir] [golden_root]
#     period_ns    default 4.0 (250 MHz)
#     report_dir   default synth_reports_kv260, under this folder
#     golden_root  folder with the LUT tables, default ../goldens/itmn_m2
# SYNTH_ONLY set in the environment stops after synthesis.

set VARIANT_DIR [file dirname [file normalize [info script]]]
set TOP         Mamba_Top
set PART        xck26-sfvc784-2LV-c

set CLK_PERIOD_NS [expr {$argc > 0 ? [lindex $argv 0] : 4.0}]
set REPORT_DIR    [file join $VARIANT_DIR [expr {$argc > 1 ? [lindex $argv 1] : "synth_reports_kv260"}]]
set GOLDEN_ROOT   [expr {$argc > 2 ? [lindex $argv 2] : [file normalize [file join $VARIANT_DIR .. goldens itmn_m2]]}]
set FREQ_MHZ      [format "%.1f" [expr {1000.0 / $CLK_PERIOD_NS}]]
file mkdir $REPORT_DIR

set SRC_V {}
foreach f {common/BRAM_256b.v common/Exp_LUT.v common/Silu_LUT.v common/Softplus_LUT.v common/RSqrt_ROM.v
           common/RSqrt_FP_ROM.v common/LUT_Bank.v common/Memory_Cluster.v mamba/H_RegFile.v mamba/Reduce16Wide.v
           mamba/Mamba_PE_v2.v mamba/M_Cluster.v mamba/Mamba_Cluster.v top/Mamba_Top.v} {
    lappend SRC_V [file join $VARIANT_DIR src $f]
}

puts "SYNTH $TOP on $PART at $FREQ_MHZ MHz ($CLK_PERIOD_NS ns), tables from $GOLDEN_ROOT"
read_verilog $SRC_V

file delete -force [file join $VARIANT_DIR .Xil]
# Serial start-up: the parallel prep step intermittently fails to read its own files on this machine
set_param general.maxThreads 1

synth_design -top $TOP -part $PART -mode out_of_context -flatten_hierarchy rebuilt \
    -include_dirs [file join $VARIANT_DIR src common] \
    -verilog_define "GOLDEN_ROOT=\"$GOLDEN_ROOT\""

create_clock -name clk -period $CLK_PERIOD_NS [get_ports clk]
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]
# The DMA buses belong to the host interconnect, the configuration inputs are static during a run
foreach p {rst start dma_write_en dma_target dma_addr dma_wdata dma_read_en dma_rtarget dma_raddr
           T_MAX CH_OUT CH_M PHASE_INC CH_IN FIN_EN POOL_OUT ENC_EN GAP_EN GAP_INV} {
    set_false_path -from [get_ports $p]
}
set_false_path -to [get_ports dma_rdata]

write_checkpoint -force [file join $REPORT_DIR post_synth.dcp]
report_utilization    -file [file join $REPORT_DIR util_synth.rpt]
report_timing_summary -max_paths 10 -file [file join $REPORT_DIR timing_synth.rpt]
report_timing -max_paths 50 -nworst 1 -unique_pins -file [file join $REPORT_DIR timing_synth_top50.rpt]
if {[info exists ::env(SYNTH_ONLY)]} { puts "SYNTH_ONLY: skipping implementation"; exit }

opt_design
place_design    -directive Explore
phys_opt_design -directive Explore
route_design    -directive Explore
phys_opt_design -directive AggressiveExplore
phys_opt_design -hold_fix

write_checkpoint -force [file join $REPORT_DIR post_route.dcp]
report_timing_summary -max_paths 20 -file [file join $REPORT_DIR timing_route.rpt]
report_timing -nworst 20 -delay_type max -slack_less_than 0.0 -file [file join $REPORT_DIR timing_violations.rpt]
report_utilization -file [file join $REPORT_DIR util_route.rpt]
report_power       -file [file join $REPORT_DIR power_route.rpt]

set util_str  [report_utilization    -return_string]
set time_str  [report_timing_summary -return_string]
set power_str [report_power          -return_string]
puts "RESULT $TOP at $FREQ_MHZ MHz ($CLK_PERIOD_NS ns)"
if {[regexp {WNS\(ns\)\s+([-\d.]+)} $time_str -> wns]} {
    puts [format "  WNS   : %+.3f ns (%s), fmax at WNS 0: %.1f MHz" $wns [expr {$wns >= 0 ? "MET" : "VIOLATED"}] \
          [expr {1000.0 / ($CLK_PERIOD_NS - $wns)}]]
}
foreach {label pat} {LUT {CLB LUTs\s*\|\s*(\d+)} FF {CLB Registers\s*\|\s*(\d+)} BRAM {Block RAM Tile\s*\|\s*(\d+)}
                     URAM {UltraRAM\s*\|\s*(\d+)} DSP {DSPs\s*\|\s*(\d+)}} {
    if {[regexp $pat $util_str -> val]} { puts [format "  %-6s: %s" $label $val] }
}
if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([\d.]+)} $power_str -> pwr]} { puts "  Power : $pwr W" }
puts "  Reports: $REPORT_DIR"
