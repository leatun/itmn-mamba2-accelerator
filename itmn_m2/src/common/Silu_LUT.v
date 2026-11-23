`include "_parameter.v"
`include "goldens.vh"

// SiLU(x), 256 entries over [-8, 8) in Q4.11; below the range 0, above it x.
module Silu_LUT (
    input  signed [15:0] x_in,
    output signed [15:0] silu_out
);

    localparam signed [15:0] LUT_LO    = -(16'sd1 << (`FRAC_BITS + 3));
    localparam        [3:0]  LUT_SHIFT = `FRAC_BITS - 4'd4;

    wire       in_range = (x_in >= LUT_LO) && (x_in < -LUT_LO);
    wire [7:0] idx      = in_range ? ($signed(x_in - LUT_LO) >>> LUT_SHIFT) : 8'd0;

    reg signed [15:0] silu_table [0:255];
    initial $readmemh({`GOLDEN_ROOT, "/silu_lut.txt"}, silu_table);

    assign silu_out = in_range        ? silu_table[idx] :
                      (x_in < LUT_LO) ? 16'sd0 : x_in;

endmodule
