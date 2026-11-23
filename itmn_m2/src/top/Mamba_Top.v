`include "_parameter.v"

// Two clusters, each with its own Memory_Cluster. Main-RAM writes to the mirrored regions are copied into the
// peer's RAM, so both hold the whole mixer area and the bulk data. The stage barrier releases both clusters
// together once both are done and no mirrored write is in flight.
//
// DMA targets (write / read): 0 main RAM (written to both), 2 cluster 0 weights, 3 const RAM (written to both),
// 4 cluster 1 weights. Reads of 0 and 3 come from cluster 0.
module Mamba_Top #(
    parameter H_ADDR_W = `H_ADDR_W,
    parameter H_DEPTH  = `H_DEPTH
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    output wire                          done_stage,
    output wire                          done_all,

    input  wire [9:0]                    T_MAX,
    input  wire [8:0]                    CH_OUT,
    input  wire [8:0]                    CH_M,
    input  wire                          PHASE_INC,
    input  wire [8:0]                    CH_IN,
    input  wire                          FIN_EN,
    input  wire                          POOL_OUT,
    input  wire                          ENC_EN,
    input  wire                          GAP_EN,
    input  wire [15:0]                   GAP_INV,

    input  wire                          dma_write_en,
    input  wire [2:0]                    dma_target,
    input  wire [14:0]                   dma_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] dma_wdata,
    input  wire                          dma_read_en,
    input  wire [2:0]                    dma_rtarget,
    input  wire [14:0]                   dma_raddr,
    output wire [`NUM_LANES*`DATA_W-1:0] dma_rdata
);

    wire [14:0]                   c0_m_rd_addr,     c1_m_rd_addr;
    wire [`NUM_LANES*`DATA_W-1:0] c0_m_rd_data,     c1_m_rd_data;
    wire                          c0_m_we,          c1_m_we;
    wire [14:0]                   c0_m_wr_addr,     c1_m_wr_addr;
    wire [`NUM_LANES*`DATA_W-1:0] c0_m_wr_data,     c1_m_wr_data;
    wire [14:0]                   c0_w_rd_addr,     c1_w_rd_addr;
    wire [14:0]                   c0_w_rd_addr2,    c1_w_rd_addr2;
    wire [`NUM_LANES*`DATA_W-1:0] c0_w_rd_data,     c1_w_rd_data;
    wire [`NUM_LANES*`DATA_W-1:0] c0_w_rd_data2,    c1_w_rd_data2;
    wire [14:0]                   c0_wx_rd_addr,    c1_wx_rd_addr;
    wire [14:0]                   c0_wx_rd_addr2,   c1_wx_rd_addr2;
    wire [`NUM_LANES*`DATA_W-1:0] c0_wx_rd_data,    c1_wx_rd_data;
    wire [`NUM_LANES*`DATA_W-1:0] c0_wx_rd_data2,   c1_wx_rd_data2;
    wire [14:0]                   c0_const_rd_addr, c1_const_rd_addr;
    wire [`NUM_LANES*`DATA_W-1:0] c0_const_rd_data, c1_const_rd_data;
    wire                          c0_done_stage,    c1_done_stage;
    wire                          c0_done_all,      c1_done_all;
    wire [44:0]                   c0_gn_part,       c1_gn_part;

    // [PT_MIX_END, PT_INPUT) is unused, everything else is mirrored
    function mirror_region;
        input [14:0] addr;
        mirror_region = (addr < `PT_MIX_END) || (addr >= `PT_INPUT);
    endfunction
    wire mirror_c0_wr = c0_m_we && mirror_region(c0_m_wr_addr);
    wire mirror_c1_wr = c1_m_we && mirror_region(c1_m_wr_addr);

    // Mirrored writes in flight, counted from the cluster's write port to the peer RAM's write (direct or from its
    // queue). The barrier waits for both counts to reach 0 in the current cycle, so a stage never reads a peer word
    // before it has landed, whatever the latency of the mirror path.
    wire       mem0_xc_land, mem1_xc_land;
    reg  [4:0] xc_pend0, xc_pend1;                         // c0 -> mem1, c1 -> mem0
    wire [4:0] xc_pend0_n = xc_pend0 + {4'd0, mirror_c0_wr} - {4'd0, mem1_xc_land};
    wire [4:0] xc_pend1_n = xc_pend1 + {4'd0, mirror_c1_wr} - {4'd0, mem0_xc_land};
    always @(posedge clk or posedge rst)
        if (rst) begin
            xc_pend0 <= 5'd0;
            xc_pend1 <= 5'd0;
        end else begin
            xc_pend0 <= xc_pend0_n;
            xc_pend1 <= xc_pend1_n;
        end
    wire bar_go = c0_done_stage && c1_done_stage && (xc_pend0_n == 5'd0) && (xc_pend1_n == 5'd0);

    // synthesis translate_off
    // The 5-bit counts hold the 16-entry queues; a deeper queue or a pipelined mirror path needs wider counters.
    always @(posedge clk)
        if (!rst && (xc_pend0_n > 5'd16 || xc_pend1_n > 5'd16 ||
                     (xc_pend0 == 5'd0 && mem1_xc_land && !mirror_c0_wr) ||
                     (xc_pend1 == 5'd0 && mem0_xc_land && !mirror_c1_wr))) begin
            $display("ERROR: Mamba_Top: mirrored-write count out of range (%0d, %0d)", xc_pend0_n, xc_pend1_n);
            $stop;
        end
    // synthesis translate_on

    wire dma_main_wr  = dma_write_en && (dma_target == 3'd0);
    wire dma_const_wr = dma_write_en && (dma_target == 3'd3);
    wire dma_w0_wr    = dma_write_en && (dma_target == 3'd2);
    wire dma_w1_wr    = dma_write_en && (dma_target == 3'd4);

    wire       c0_dma_we     = dma_main_wr || dma_const_wr || dma_w0_wr;
    wire [2:0] c0_dma_target = dma_w0_wr ? 3'd2 : dma_const_wr ? 3'd3 : 3'd0;
    wire       c1_dma_we     = dma_main_wr || dma_const_wr || dma_w1_wr;
    wire [2:0] c1_dma_target = dma_w1_wr ? 3'd2 : dma_const_wr ? 3'd3 : 3'd0;

    wire       dma_rd_w1    = dma_read_en && (dma_rtarget == 3'd4);
    wire [`NUM_LANES*`DATA_W-1:0] c0_dma_rdata, c1_dma_rdata;
    assign dma_rdata = dma_rd_w1 ? c1_dma_rdata : c0_dma_rdata;

    Mamba_Cluster #(.CLUSTER_ID(0), .H_ADDR_W(H_ADDR_W), .H_DEPTH(H_DEPTH)) u_c0 (
        .clk             (clk),
        .rst             (rst),
        .start           (start),
        .bar_go          (bar_go),
        .done_stage      (c0_done_stage),
        .done_all        (c0_done_all),
        .gn_part         (c0_gn_part),
        .peer_gn_part    (c1_gn_part),
        .T_MAX           (T_MAX),
        .CH_OUT          (CH_OUT),
        .CH_M            (CH_M),
        .PHASE_INC       (PHASE_INC),
        .CH_IN           (CH_IN),
        .FIN_EN          (FIN_EN),
        .POOL_OUT        (POOL_OUT),
        .ENC_EN          (ENC_EN),
        .GAP_EN          (GAP_EN),
        .GAP_INV         (GAP_INV),
        .m_rd_addr       (c0_m_rd_addr),
        .m_rd_data_c     (c0_m_rd_data),
        .m_we            (c0_m_we),
        .m_wr_addr       (c0_m_wr_addr),
        .m_wr_data       (c0_m_wr_data),
        .w_rd_addr       (c0_w_rd_addr),
        .w_rd_addr2      (c0_w_rd_addr2),
        .w_rd_data       (c0_w_rd_data),
        .w_rd_data2      (c0_w_rd_data2),
        .wx_rd_addr      (c0_wx_rd_addr),
        .wx_rd_addr2     (c0_wx_rd_addr2),
        .wx_rd_data      (c0_wx_rd_data),
        .wx_rd_data2     (c0_wx_rd_data2),
        .const_rd_addr   (c0_const_rd_addr),
        .const_rd_data   (c0_const_rd_data)
    );

    Mamba_Cluster #(.CLUSTER_ID(1), .H_ADDR_W(H_ADDR_W), .H_DEPTH(H_DEPTH)) u_c1 (
        .clk             (clk),
        .rst             (rst),
        .start           (start),
        .bar_go          (bar_go),
        .done_stage      (c1_done_stage),
        .done_all        (c1_done_all),
        .gn_part         (c1_gn_part),
        .peer_gn_part    (c0_gn_part),
        .T_MAX           (T_MAX),
        .CH_OUT          (CH_OUT),
        .CH_M            (CH_M),
        .PHASE_INC       (PHASE_INC),
        .CH_IN           (CH_IN),
        .FIN_EN          (FIN_EN),
        .POOL_OUT        (POOL_OUT),
        .ENC_EN          (ENC_EN),
        .GAP_EN          (GAP_EN),
        .GAP_INV         (GAP_INV),
        .m_rd_addr       (c1_m_rd_addr),
        .m_rd_data_c     (c1_m_rd_data),
        .m_we            (c1_m_we),
        .m_wr_addr       (c1_m_wr_addr),
        .m_wr_data       (c1_m_wr_data),
        .w_rd_addr       (c1_w_rd_addr),
        .w_rd_addr2      (c1_w_rd_addr2),
        .w_rd_data       (c1_w_rd_data),
        .w_rd_data2      (c1_w_rd_data2),
        .wx_rd_addr      (c1_wx_rd_addr),
        .wx_rd_addr2     (c1_wx_rd_addr2),
        .wx_rd_data      (c1_wx_rd_data),
        .wx_rd_data2     (c1_wx_rd_data2),
        .const_rd_addr   (c1_const_rd_addr),
        .const_rd_data   (c1_const_rd_data)
    );

    Memory_Cluster u_mem0 (
        .clk           (clk),
        .reset         (rst),
        .core_rd_addr  (c0_m_rd_addr),
        .core_rd_data  (c0_m_rd_data),
        .core_we       (c0_m_we),
        .core_wr_addr  (c0_m_wr_addr),
        .core_wr_data  (c0_m_wr_data),
        .xc_we         (mirror_c1_wr),
        .xc_wr_addr    (c1_m_wr_addr),
        .xc_wr_data    (c1_m_wr_data),
        .xc_land       (mem0_xc_land),
        .w_rd_addr     (c0_w_rd_addr),
        .w_rd_data     (c0_w_rd_data),
        .w_rd_addr2    (c0_w_rd_addr2),
        .w_rd_data2    (c0_w_rd_data2),
        .wx_rd_addr    (c0_wx_rd_addr),
        .wx_rd_data    (c0_wx_rd_data),
        .wx_rd_addr2   (c0_wx_rd_addr2),
        .wx_rd_data2   (c0_wx_rd_data2),
        .const_rd_addr (c0_const_rd_addr),
        .const_rd_data (c0_const_rd_data),
        .dma_write_en  (c0_dma_we),
        .dma_target    (c0_dma_target),
        .dma_addr      (dma_addr),
        .dma_wdata     (dma_wdata),
        .dma_read_en   (dma_read_en && !dma_rd_w1),
        .dma_rtarget   (dma_rtarget),
        .dma_raddr     (dma_raddr),
        .dma_rdata     (c0_dma_rdata)
    );

    Memory_Cluster u_mem1 (
        .clk           (clk),
        .reset         (rst),
        .core_rd_addr  (c1_m_rd_addr),
        .core_rd_data  (c1_m_rd_data),
        .core_we       (c1_m_we),
        .core_wr_addr  (c1_m_wr_addr),
        .core_wr_data  (c1_m_wr_data),
        .xc_we         (mirror_c0_wr),
        .xc_wr_addr    (c0_m_wr_addr),
        .xc_wr_data    (c0_m_wr_data),
        .xc_land       (mem1_xc_land),
        .w_rd_addr     (c1_w_rd_addr),
        .w_rd_data     (c1_w_rd_data),
        .w_rd_addr2    (c1_w_rd_addr2),
        .w_rd_data2    (c1_w_rd_data2),
        .wx_rd_addr    (c1_wx_rd_addr),
        .wx_rd_data    (c1_wx_rd_data),
        .wx_rd_addr2   (c1_wx_rd_addr2),
        .wx_rd_data2   (c1_wx_rd_data2),
        .const_rd_addr (c1_const_rd_addr),
        .const_rd_data (c1_const_rd_data),
        .dma_write_en  (c1_dma_we),
        .dma_target    (c1_dma_target),
        .dma_addr      (dma_addr),
        .dma_wdata     (dma_wdata),
        .dma_read_en   (dma_rd_w1),
        .dma_rtarget   (3'd2),
        .dma_raddr     (dma_raddr),
        .dma_rdata     (c1_dma_rdata)
    );

    assign done_all   = c0_done_all & c1_done_all;
    assign done_stage = done_all;

endmodule
