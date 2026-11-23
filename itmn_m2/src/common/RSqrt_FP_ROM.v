`timescale 1ns / 1ps
`include "_parameter.v"
`include "goldens.vh"

// Gated RMSNorm rsqrt on a normalised mantissa: rom[i] = min(32767, round(2^21 / sqrt(8i + 4))), i = (m >> 3) - 1024,
// where m is the sum of squares (+ eps) shifted by an even amount into [2^13, 2^15). The controller applies the
// output shift; the sqrt(2) of an odd log2(d_inner) is folded into gamma offline.
module RSqrt_FP_ROM (
    input  [11:0] idx,
    output [15:0] data
);

    reg [15:0] rom [0:3071];
    initial $readmemh({`GOLDEN_ROOT, "/rsqrt_fp.txt"}, rom);

    assign data = rom[idx];

endmodule
