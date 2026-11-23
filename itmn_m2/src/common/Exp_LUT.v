`include "_parameter.v"
`include "goldens.vh"

// exp(x), 256 entries over [-8, 8) in Q4.11; below the range 0, above it saturated.
module Exp_LUT (
    input  signed [15:0] x_in,
    output signed [15:0] exp_out
);

    localparam signed [15:0] LUT_LO    = -(16'sd1 << (`FRAC_BITS + 3));
    localparam        [3:0]  LUT_SHIFT = `FRAC_BITS - 4'd4;

    wire       in_range = (x_in >= LUT_LO) && (x_in < -LUT_LO);
    wire [7:0] idx      = in_range ? ($signed(x_in - LUT_LO) >>> LUT_SHIFT) : 8'd0;

    reg signed [15:0] exp_table [0:255];
    initial $readmemh({`GOLDEN_ROOT, "/exp_lut.txt"}, exp_table);

    assign exp_out = in_range         ? exp_table[idx] :
                     (x_in < LUT_LO)  ? 16'sd0 : 16'sh7FFF;

endmodule
