`include "_parameter.v"

// 16 PEs sharing op_mode / clear_acc, plus the scan state RAM. With h_from_rf the PEs' H operand is the RAM word
// read at h_rd_addr three cycles earlier (RAM read + output register), else in_H_ext.
module M_Cluster #(
    parameter H_ADDR_W = `H_ADDR_W,
    parameter H_DEPTH  = `H_DEPTH
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [2:0]                    op_mode,
    input  wire                          clear_acc,
    input  wire [`NUM_LANES*`DATA_W-1:0] in_W1_vec,
    input  wire [`NUM_LANES*`DATA_W-1:0] in_H_ext,
    input  wire [`NUM_LANES*`DATA_W-1:0] in_W2_vec,
    input  wire [`NUM_LANES*`DATA_W-1:0] in_X_vec,
    input  wire                          h_from_rf,
    input  wire [H_ADDR_W-1:0]           h_rd_addr,
    input  wire                          h_wr_en,
    input  wire [H_ADDR_W-1:0]           h_wr_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] h_wr_data,
    output wire [`NUM_LANES*`DATA_W-1:0] out_vec,
    output wire [`NUM_LANES*`ACC_W-1:0]  acc_raw_vec,
    output wire [`NUM_LANES*`DATA_W-1:0] out_vec2
);

    wire [`NUM_LANES*`DATA_W-1:0] h_rd_data;
    reg  [`NUM_LANES*`DATA_W-1:0] h_rd_data_r;

    H_RegFile #(.ADDR_W(H_ADDR_W), .DEPTH(H_DEPTH)) u_hrf (
        .clk     (clk),
        .rst     (rst),
        .wr_en   (h_wr_en),
        .wr_addr (h_wr_addr),
        .wr_data (h_wr_data),
        .rd_addr (h_rd_addr),
        .rd_data (h_rd_data)
    );

    always @(posedge clk or posedge rst)
        if (rst) h_rd_data_r <= {`NUM_LANES*`DATA_W{1'b0}};
        else     h_rd_data_r <= h_rd_data;

    wire [`NUM_LANES*`DATA_W-1:0] in_H_vec = h_from_rf ? h_rd_data_r : in_H_ext;

    genvar i;
    generate
        for (i = 0; i < `NUM_LANES; i = i + 1) begin : PE_LANE
            Mamba_PE_v2 u_pe (
                .clk       (clk),
                .rst       (rst),
                .op_mode   (op_mode),
                .clear_acc (clear_acc),
                .in_W1     (in_W1_vec  [i*`DATA_W +: `DATA_W]),
                .in_H      (in_H_vec   [i*`DATA_W +: `DATA_W]),
                .in_W2     (in_W2_vec  [i*`DATA_W +: `DATA_W]),
                .in_X      (in_X_vec   [i*`DATA_W +: `DATA_W]),
                .out_val   (out_vec    [i*`DATA_W +: `DATA_W]),
                .acc_raw   (acc_raw_vec[i*`ACC_W  +: `ACC_W ]),
                .out_val2  (out_vec2   [i*`DATA_W +: `DATA_W])
            );
        end
    endgenerate

endmodule
