`include "_parameter.v"

// 16 lanes each of SiLU, softplus and exp (one table copy per lane: one read port each), plus the RMSNorm rsqrt ROM.
module LUT_Bank (
    input  wire [`NUM_LANES*`DATA_W-1:0] silu_in_flat,
    input  wire [`NUM_LANES*`DATA_W-1:0] sp_in_flat,
    input  wire [`NUM_LANES*`DATA_W-1:0] exp_in_flat,
    output wire [`NUM_LANES*`DATA_W-1:0] silu_out_flat,
    output wire [`NUM_LANES*`DATA_W-1:0] sp_out_flat,
    output wire [`NUM_LANES*`DATA_W-1:0] exp_out_flat,
    input  wire [12:0]                   rsqrt_idx,
    output wire [`DATA_W-1:0]            rsqrt_data
);

    genvar gi;
    generate
        for (gi = 0; gi < `NUM_LANES; gi = gi + 1) begin : ACT_LANES
            Silu_LUT u_silu (
                .x_in     (silu_in_flat [gi*`DATA_W +: `DATA_W]),
                .silu_out (silu_out_flat[gi*`DATA_W +: `DATA_W])
            );
            Softplus_LUT u_softplus (
                .x_in         (sp_in_flat [gi*`DATA_W +: `DATA_W]),
                .softplus_out (sp_out_flat[gi*`DATA_W +: `DATA_W])
            );
            Exp_LUT u_exp (
                .x_in    (exp_in_flat [gi*`DATA_W +: `DATA_W]),
                .exp_out (exp_out_flat[gi*`DATA_W +: `DATA_W])
            );
        end
    endgenerate

    RSqrt_ROM u_rsqrt_rom (
        .idx  (rsqrt_idx),
        .data (rsqrt_data)
    );

endmodule
