`include "_parameter.v"

// Scan state RAM, one 16-lane word per (channel, state word). No initial block, so it can map to URAM; the
// controller zeroes it (S_H_INIT) before every block.
module H_RegFile #(
    parameter ADDR_W = `H_ADDR_W,
    parameter DEPTH  = `H_DEPTH
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          wr_en,
    input  wire [ADDR_W-1:0]             wr_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] wr_data,
    input  wire [ADDR_W-1:0]             rd_addr,
    output reg  [`NUM_LANES*`DATA_W-1:0] rd_data
);

    (* ram_style = "ultra" *)
    reg [`NUM_LANES*`DATA_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    always @(posedge clk or posedge rst)
        if (rst) rd_data <= {`NUM_LANES*`DATA_W{1'b0}};
        else     rd_data <= mem[rd_addr];

endmodule
