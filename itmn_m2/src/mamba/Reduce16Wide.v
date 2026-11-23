`include "_parameter.v"

// Sum of 16 signed 40-bit lanes, exact (45-bit result), one register in the middle of the tree.
module Reduce16Wide (
    input  wire                                clk,
    input  wire                                rst,
    input  wire signed [`NUM_LANES*`ACC_W-1:0] in_vec,
    output wire signed [`ACC_W+4:0]            out_sum
);

    wire signed [`ACC_W-1:0] lane [0:`NUM_LANES-1];
    genvar i;
    generate
        for (i = 0; i < `NUM_LANES; i = i + 1) begin : UNPACK
            assign lane[i] = in_vec[i*`ACC_W +: `ACC_W];
        end
    endgenerate

    wire signed [`ACC_W:0]   s0_0 = lane[0]  + lane[1];
    wire signed [`ACC_W:0]   s0_1 = lane[2]  + lane[3];
    wire signed [`ACC_W:0]   s0_2 = lane[4]  + lane[5];
    wire signed [`ACC_W:0]   s0_3 = lane[6]  + lane[7];
    wire signed [`ACC_W:0]   s0_4 = lane[8]  + lane[9];
    wire signed [`ACC_W:0]   s0_5 = lane[10] + lane[11];
    wire signed [`ACC_W:0]   s0_6 = lane[12] + lane[13];
    wire signed [`ACC_W:0]   s0_7 = lane[14] + lane[15];

    reg  signed [`ACC_W+1:0] s1_0_r, s1_1_r, s1_2_r, s1_3_r;
    always @(posedge clk or posedge rst)
        if (rst) begin
            s1_0_r <= {(`ACC_W+2){1'b0}};
            s1_1_r <= {(`ACC_W+2){1'b0}};
            s1_2_r <= {(`ACC_W+2){1'b0}};
            s1_3_r <= {(`ACC_W+2){1'b0}};
        end else begin
            s1_0_r <= s0_0 + s0_1;
            s1_1_r <= s0_2 + s0_3;
            s1_2_r <= s0_4 + s0_5;
            s1_3_r <= s0_6 + s0_7;
        end

    wire signed [`ACC_W+2:0] s2_0 = s1_0_r + s1_1_r;
    wire signed [`ACC_W+2:0] s2_1 = s1_2_r + s1_3_r;
    wire signed [`ACC_W+3:0] s3   = s2_0 + s2_1;

    assign out_sum = {s3[`ACC_W+3], s3};

endmodule
