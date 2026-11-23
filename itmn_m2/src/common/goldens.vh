// Root of the golden / table files read with $readmemh (LUT tables, testbench data).
// Relative to the simulator's working directory (the variant folder); override with
// +define+GOLDEN_ROOT=\"<path>\" (the synth script passes an absolute path).
`ifndef GOLDEN_ROOT
  `define GOLDEN_ROOT "../goldens/itmn_m2"
`endif
