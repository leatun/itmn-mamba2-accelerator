`timescale 1ns / 1ps
`include "_parameter.v"
`include "goldens.vh"

// Input RMSNorm rsqrt: rom[m] = round(sqrt(2^(1+N)) * SCALE / sqrt(m)), N = 6 (rom[128] = 1.0 in Q4.11).
module RSqrt_ROM (
    input  [12:0] idx,
    output [15:0] data
);

    reg [15:0] rom [0:8191];
    initial $readmemh({`GOLDEN_ROOT, "/rsqrt_q97.txt"}, rom);

    assign data = rom[idx];

endmodule
