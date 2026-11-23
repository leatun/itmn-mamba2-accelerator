`include "_parameter.v"

// One of the two clusters of the ITMN-Mamba2 accelerator: controller FSM, a 16-lane PE array with the scan state
// RAM (M_Cluster), activation LUTs, a lane-reduce tree and the fabric multiplier rows of the scan. Mamba_Top wires
// each cluster to its own Memory_Cluster and mirrors shared writes to the peer.
//
// Work split: the Mamba phase by 16-channel group (cluster 0 even groups, cluster 1 odd), the Inception phase by
// timestep parity. Every stage ends in S_BARRIER, which releases (bar_go) when both clusters have arrived and every
// write mirrored into the peer's RAM has landed there.
//
// Mamba phase, per timestep t:
//   RN   input RMSNorm. For t >= 1 cluster 1 already computed it at the end of in_proj of t-1 (X_NORM is
//        double-buffered by t parity), so the stage is only a barrier.
//   M1   in_proj -> [z | xBC | dt]
//   M2   causal conv (width 4) + bias, SiLU on the write
//   M6   scan: dt = softplus(dt_raw + bias), dA = exp(dt*A) in the load phase, then one state word per cycle;
//        y = C.h + D*x, y_gated = y * SiLU(z)
//   GN   gated RMSNorm
//   M8   out_proj -> PT_MAMBA_OUT[t]
// After the last t: FIN (combine with the Inception output, MaxPool2, GAP accumulate), then GAP and FC on block 4.
// Inception phase: encoder (block 0), P1, MP3, B1, BOT, B2..B4, all on the MAC chain except MP3.
// Every MAC job is split over its input words (split-K): the PE array takes the first half, the scan's two fabric
// rows the second half (idle outside M6), and the two exact partial sums are added before the one shift.
//
// Read latencies seen by the FSM: main RAM 3 cycles after the address, const RAM 2, weight RAM 3. PE results
// (out_val) are valid 6 cycles after the op is assigned, acc_raw 5.
module Mamba_Cluster #(
    parameter CLUSTER_ID = 0,
    parameter H_ADDR_W   = `H_ADDR_W,
    parameter H_DEPTH    = `H_DEPTH
)(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,

    input  wire         bar_go,         // both clusters in S_BARRIER, no mirrored write in flight (Mamba_Top)
    output wire         done_stage,     // in S_BARRIER with every write issued
    output wire         done_all,
    // GN: each cluster's share of sum(y_gated^2), summed on the scan's gate row; final while in the barrier
    output wire [44:0]  gn_part,
    input  wire [44:0]  peer_gn_part,

    // Workload, stable from before start until done
    input  wire [9:0]   T_MAX,
    input  wire [8:0]   CH_OUT,         // d_model / 16
    input  wire [8:0]   CH_M,           // d_inner / 16
    input  wire         PHASE_INC,      // 1: Inception phase, 0: Mamba phase
    input  wire [8:0]   CH_IN,          // block input / 16 (P1), or encoder output groups
    input  wire         FIN_EN,         // run FIN after the t-loop
    input  wire         POOL_OUT,       // FIN output MaxPool1d(2, 2)
    input  wire         ENC_EN,         // Inception phase starts with the encoder (block 0)
    input  wire         GAP_EN,         // GAP over T and FC head after FIN (block 4)
    input  wire [15:0]  GAP_INV,        // round(2^15 / T)

    output reg  [14:0]  m_rd_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] m_rd_data_c,
    output reg          m_we,
    output reg  [14:0]  m_wr_addr,
    output reg  [`NUM_LANES*`DATA_W-1:0] m_wr_data,
    output reg  [14:0]  w_rd_addr,
    output reg  [14:0]  w_rd_addr2,
    input  wire [`NUM_LANES*`DATA_W-1:0] w_rd_data,
    input  wire [`NUM_LANES*`DATA_W-1:0] w_rd_data2,
    output reg  [14:0]  wx_rd_addr,
    output reg  [14:0]  wx_rd_addr2,
    input  wire [`NUM_LANES*`DATA_W-1:0] wx_rd_data,
    input  wire [`NUM_LANES*`DATA_W-1:0] wx_rd_data2,
    output reg  [14:0]  const_rd_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] const_rd_data
);

    localparam [14:0] PT_X_NORM        = `PT_X_NORM;
    localparam [14:0] PT_X_NORM_B      = `PT_X_NORM_B;
    localparam [14:0] PT_INP_FRAMES    = `PT_INP_FRAMES;
    localparam [14:0] PT_XCONV         = `PT_XCONV;
    localparam [14:0] PT_XBC           = `PT_XBC;
    localparam [14:0] PT_DT            = `PT_DT;
    localparam [14:0] PT_DA            = `PT_DA;
    localparam [14:0] PT_Y_SSM         = `PT_Y_SSM;
    localparam [14:0] PT_Y_GATED       = `PT_Y_GATED;
    localparam [14:0] PT_Y_NORM        = `PT_Y_NORM;
    localparam [14:0] PT_INPUT         = `PT_INPUT;
    localparam [14:0] PT_MAMBA_OUT     = `PT_MAMBA_OUT;
    localparam [14:0] PT_X_IN          = `PT_X_IN;
    localparam [14:0] PT_INC_OUT       = `PT_INC_OUT;
    localparam [14:0] PT_MP3           = `PT_MP3;
    localparam [14:0] PT_BOT           = `PT_BOT;
    localparam [14:0] PT_ENC_IN        = `PT_ENC_IN;
    localparam [14:0] PT_GAP           = `PT_GAP;
    localparam [14:0] PT_LOGITS        = `PT_LOGITS;
    localparam [14:0] W_CONV_BASE      = `W_CONV_BASE;
    localparam [14:0] W_INPROJ_BASE    = `W_INPROJ_BASE;
    localparam [14:0] W_OUTPROJ_BASE   = `W_OUTPROJ_BASE;
    localparam [14:0] W_P1_BASE        = `W_P1_BASE;
    localparam [14:0] W_BOT_BASE       = `W_BOT_BASE;
    localparam [14:0] W_B1_BASE        = `W_B1_BASE;
    localparam [14:0] W_B2_BASE        = `W_B2_BASE;
    localparam [14:0] W_B3_BASE        = `W_B3_BASE;
    localparam [14:0] W_B4_BASE        = `W_B4_BASE;
    localparam [14:0] W_ENC_BASE       = `W_ENC_BASE;
    localparam [14:0] W_FC_BASE        = `W_FC_BASE;
    localparam [14:0] C_W_NORM_BASE    = `C_W_NORM_BASE;
    localparam [14:0] C_B_CONV_BASE    = `C_B_CONV_BASE;
    localparam [14:0] C_DT_BIAS_BASE   = `C_DT_BIAS_BASE;
    localparam [14:0] C_A_BASE         = `C_A_BASE;
    localparam [14:0] C_D_BASE         = `C_D_BASE;
    localparam [14:0] C_W_GNORM_BASE   = `C_W_GNORM_BASE;
    localparam [14:0] C_P1_BIAS_BASE   = `C_P1_BIAS_BASE;
    localparam [14:0] C_INC_SCALE_BASE = `C_INC_SCALE_BASE;
    localparam [14:0] C_INC_SHIFT_BASE = `C_INC_SHIFT_BASE;
    localparam [14:0] C_ENC_BIAS_BASE  = `C_ENC_BIAS_BASE;
    localparam [14:0] C_FC_BIAS_BASE   = `C_FC_BIAS_BASE;

    localparam [4:0] STG_M1A = `STG_M1A;
    localparam [4:0] STG_M2  = `STG_M2;
    localparam [4:0] STG_M6  = `STG_M6;
    localparam [4:0] STG_M8  = `STG_M8;
    localparam [4:0] STG_RN  = `STG_RN;
    localparam [4:0] STG_GN  = `STG_GN;
    localparam [4:0] STG_FIN = `STG_FIN;
    localparam [4:0] STG_GAP = `STG_GAP;
    localparam [4:0] STG_FC  = `STG_FC;
    localparam [4:0] STG_P1  = `STG_P1;
    localparam [4:0] STG_MP3 = `STG_MP3;
    localparam [4:0] STG_B1  = `STG_B1;
    localparam [4:0] STG_BOT = `STG_BOT;
    localparam [4:0] STG_B2  = `STG_B2;
    localparam [4:0] STG_B3  = `STG_B3;
    localparam [4:0] STG_B4  = `STG_B4;
    localparam [4:0] STG_ENC = `STG_ENC;

    // The testbenches' profilers rely on the codes of S_IDLE, S_H_INIT, S_BARRIER and S_DONE.
    localparam [6:0] S_IDLE    = 7'd0;
    localparam [6:0] S_H_INIT  = 7'd1;
    localparam [6:0] S_MAC1    = 7'd2;
    localparam [6:0] S_MAC2    = 7'd3;
    localparam [6:0] S_MAC3    = 7'd4;
    localparam [6:0] S_MAC4    = 7'd5;
    localparam [6:0] S_MAC5    = 7'd6;
    localparam [6:0] S_RN1     = 7'd7;
    localparam [6:0] S_RN2     = 7'd8;
    localparam [6:0] S_RN3     = 7'd9;
    localparam [6:0] S_RN4     = 7'd10;
    localparam [6:0] S_RN5     = 7'd11;
    localparam [6:0] S_RN6     = 7'd12;
    localparam [6:0] S_RN7     = 7'd13;
    localparam [6:0] S_CONV0   = 7'd14;
    localparam [6:0] S_CONV    = 7'd15;
    localparam [6:0] S_SCAN_LD = 7'd16;
    localparam [6:0] S_SCAN    = 7'd17;
    localparam [6:0] S_GN1     = 7'd18;
    localparam [6:0] S_GN2     = 7'd19;
    localparam [6:0] S_GN3     = 7'd20;
    localparam [6:0] S_GN4     = 7'd21;
    localparam [6:0] S_GN5     = 7'd22;
    localparam [6:0] S_GN6     = 7'd23;
    localparam [6:0] S_MP3     = 7'd28;
    localparam [6:0] S_FIN     = 7'd29;
    localparam [6:0] S_GAP     = 7'd30;
    localparam [6:0] S_BARRIER = 7'd120;
    localparam [6:0] S_DONE    = 7'd127;

    // ------------------------------------------------------------------------
    // Lane-wise helpers
    // ------------------------------------------------------------------------
    function [`NUM_LANES*`DATA_W-1:0] sat_add_vec;
        input [`NUM_LANES*`DATA_W-1:0] a, b;
        integer i;
        reg signed [`DATA_W:0] s;
        begin
            for (i = 0; i < `NUM_LANES; i = i + 1) begin
                s = $signed(a[i*`DATA_W +: `DATA_W]) + $signed(b[i*`DATA_W +: `DATA_W]);
                sat_add_vec[i*`DATA_W +: `DATA_W] =
                    (s > 17'sd32767)  ? 16'h7FFF :
                    (s < -17'sd32768) ? 16'h8000 : s[`DATA_W-1:0];
            end
        end
    endfunction
    function automatic [`DATA_W-1:0] sat_add16;
        input [`DATA_W-1:0] a, b;
        reg   [`DATA_W:0]   s;
        begin
            s = {a[`DATA_W-1], a} + {b[`DATA_W-1], b};
            sat_add16 = (s[`DATA_W] != s[`DATA_W-1]) ? (s[`DATA_W] ? `SAT16_MIN : `SAT16_MAX) : s[`DATA_W-1:0];
        end
    endfunction
    function [`NUM_LANES*`DATA_W-1:0] max_vec;
        input [`NUM_LANES*`DATA_W-1:0] a, b;
        integer i;
        for (i = 0; i < `NUM_LANES; i = i + 1)
            max_vec[i*`DATA_W +: `DATA_W] =
                ($signed(a[i*`DATA_W +: `DATA_W]) > $signed(b[i*`DATA_W +: `DATA_W]))
                ? a[i*`DATA_W +: `DATA_W] : b[i*`DATA_W +: `DATA_W];
    endfunction
    function [`NUM_LANES*`DATA_W-1:0] relu_vec;
        input [`NUM_LANES*`DATA_W-1:0] a;
        integer i;
        for (i = 0; i < `NUM_LANES; i = i + 1)
            relu_vec[i*`DATA_W +: `DATA_W] = a[(i+1)*`DATA_W-1] ? {`DATA_W{1'b0}} : a[i*`DATA_W +: `DATA_W];
    endfunction
    function [5:0] msb45;
        input [44:0] v;
        integer i;
        begin
            msb45 = 6'd0;
            for (i = 0; i < 45; i = i + 1) if (v[i]) msb45 = i[5:0];
        end
    endfunction
    function [15:0] sat_shr40;
        input signed [`ACC_W-1:0] v;
        input [4:0] sh;
        reg signed [`ACC_W-1:0] r;
        begin
            r = v >>> sh;
            sat_shr40 = (r > 40'sd32767) ? 16'h7FFF : (r < -40'sd32768) ? 16'h8000 : r[15:0];
        end
    endfunction

    // ------------------------------------------------------------------------
    // Workload-derived values
    // ------------------------------------------------------------------------
    wire [9:0]  ch_m_act = {1'b0, CH_M};
    wire [13:0] d_inner  = {ch_m_act, 4'b0};
    wire [12:0] d_model  = {CH_OUT, 4'b0};
    wire [9:0]  t_last   = T_MAX - 10'd1;
    // in_proj output groups: z (d_inner/16), xBC (d_inner/16 + 2 per d_state word), dt (1)
    wire [9:0]  inp_g    = {ch_m_act[8:0], 1'b0} + 10'd2 * `M2_S + 10'd1;
    wire [9:0]  xbc_g    = ch_m_act + 10'd2 * `M2_S;
    // H words this cluster's scan can read: d_inner/2 channels x M2_S state words
    wire [17:0] h_init_limit = (`M2_S == 1) ? {4'b0, d_inner} : (({4'b0, d_inner} * `M2_S) >> 1);
    wire [5:0]  rn_shift = (CH_OUT == 9'd8) ? 6'd22 : 6'd21;     // 19 + log2(d_model / 16)
    wire [15:0] gn_eps   = (ch_m_act == 10'd16) ? 16'd10737 : 16'd5369;   // round(1e-5 * d_inner * 2^22)

    // in_proj output frames (4, indexed by t mod 4, the conv window). The frame-base multiply is registered: the
    // configuration is static during a run and it was the critical path.
    reg  [14:0] fb_r  [0:3];      // frame k
    reg  [14:0] fbx_r [0:3];      // its xBC part
    always @(posedge clk) begin : fb_regs
        integer k;
        for (k = 0; k < 4; k = k + 1) begin
            fb_r[k]  <= PT_INP_FRAMES + k * inp_g;
            fbx_r[k] <= PT_INP_FRAMES + k * inp_g + ch_m_act;
        end
    end

    // ------------------------------------------------------------------------
    // Registers
    // ------------------------------------------------------------------------
    (* MAX_FANOUT = 50 *) reg [6:0]  state;
    (* MAX_FANOUT = 50 *) reg [4:0]  cur_stage;
    (* MAX_FANOUT = 30 *) reg [8:0]  ctr_g;
    (* MAX_FANOUT = 30 *) reg [12:0] ctr_k;
    reg [9:0]   t_cnt;
    reg [2:0]   wait_cnt;
    reg [15:0]  h_init_cnt;
    reg [4:0]   next_stage_q;       // applied when the barrier releases
    reg [6:0]   next_state_q;
    reg [8:0]   ps;                 // slot counter of the pipelined stages (RN apply, conv, scan, GN apply)
    reg [14:0]  t_ch_r;             // t_cnt * CH_OUT in the Mamba phase, kept by an adder

    reg  [2:0]                        cl_op_mode;
    reg                               cl_clear_acc;
    reg  [`NUM_LANES*`DATA_W-1:0]     cl_in_W1_vec, cl_in_H_ext, cl_in_W2_vec, cl_in_X_vec;
    wire [`NUM_LANES*`DATA_W-1:0]     cl_out_vec, cl_out_vec2;
    wire [`NUM_LANES*`ACC_W-1:0]      cl_acc_raw_vec;

    reg                               h_from_rf_r;
    reg  [H_ADDR_W-1:0]               h_rd_addr_r;
    reg                               h_wr_en_r;
    reg  [H_ADDR_W-1:0]               h_wr_addr_r;
    reg  [`NUM_LANES*`DATA_W-1:0]     h_wr_data_r;

    // Read data registers (cut the RAM output -> FSM input paths)
    reg  [`NUM_LANES*`DATA_W-1:0]     m_rd_data;
    reg  [`NUM_LANES*`DATA_W-1:0]     w_rd_data_r, w_rd_data2_r, wx_rd_data_r, wx_rd_data2_r;
    reg                               ax_cap;          // the read now on m_rd_data_c is the aux x word
    always @(posedge clk or posedge rst)
        if (rst) begin
            m_rd_data     <= {`NUM_LANES*`DATA_W{1'b0}};
            w_rd_data_r   <= {`NUM_LANES*`DATA_W{1'b0}};
            w_rd_data2_r  <= {`NUM_LANES*`DATA_W{1'b0}};
            wx_rd_data_r  <= {`NUM_LANES*`DATA_W{1'b0}};
            wx_rd_data2_r <= {`NUM_LANES*`DATA_W{1'b0}};
        end else begin
            if (!ax_cap) m_rd_data <= m_rd_data_c;
            w_rd_data_r   <= w_rd_data;
            w_rd_data2_r  <= w_rd_data2;
            wx_rd_data_r  <= wx_rd_data;
            wx_rd_data2_r <= wx_rd_data2;
        end

    // ------------------------------------------------------------------------
    // PE array, scan state RAM, lane reduce, LUTs
    // ------------------------------------------------------------------------
    M_Cluster #(.H_ADDR_W(H_ADDR_W), .H_DEPTH(H_DEPTH)) u_mc (
        .clk         (clk),
        .rst         (rst),
        .op_mode     (cl_op_mode),
        .clear_acc   (cl_clear_acc),
        .in_W1_vec   (cl_in_W1_vec),
        .in_H_ext    (cl_in_H_ext),
        .in_W2_vec   (cl_in_W2_vec),
        .in_X_vec    (cl_in_X_vec),
        .h_from_rf   (h_from_rf_r),
        .h_rd_addr   (h_rd_addr_r),
        .h_wr_en     (h_wr_en_r),
        .h_wr_addr   (h_wr_addr_r),
        .h_wr_data   (h_wr_data_r),
        .out_vec     (cl_out_vec),
        .acc_raw_vec (cl_acc_raw_vec),
        .out_vec2    (cl_out_vec2)
    );

    // Sum of the 16 PE accumulators (sums of squares of RN and GN), valid 2 cycles after acc_raw
    wire signed [`ACC_W+4:0] sum_d_wide_c;
    reg  signed [`ACC_W+4:0] sum_d_wide;
    Reduce16Wide u_rw (
        .clk    (clk),
        .rst    (rst),
        .in_vec (cl_acc_raw_vec),
        .out_sum(sum_d_wide_c)
    );
    always @(posedge clk or posedge rst)
        if (rst) sum_d_wide <= {(`ACC_W+5){1'b0}};
        else     sum_d_wide <= sum_d_wide_c;

    reg  [`NUM_LANES*`DATA_W-1:0] dt_sum;          // dt_raw + dt_bias
    reg  [`NUM_LANES*`DATA_W-1:0] m2_res;          // conv sum + bias
    reg  [12:0]                   rsqrt_idx_r;
    wire [15:0]                   rsqrt_data;
    wire [`NUM_LANES*`DATA_W-1:0] silu_out_w, sp_out_w, exp_out_w;
    // LUT inputs are not gated by a state decode: the decode sat on the path into m_wr_data. The SiLU source flag
    // lags the state by one cycle; SiLU is read at ps >= 14 of S_CONV and ps 15 of S_SCAN, a barrier apart.
    (* max_fanout = 32 *) reg silu_src_conv;
    always @(posedge clk or posedge rst)
        if (rst) silu_src_conv <= 1'b0;
        else     silu_src_conv <= (state == S_CONV);
    wire [`NUM_LANES*`DATA_W-1:0] silu_in = silu_src_conv ? m2_res : m_rd_data;

    LUT_Bank u_lut (
        .silu_in_flat  (silu_in),
        .sp_in_flat    (dt_sum),
        .exp_in_flat   (cl_out_vec),
        .silu_out_flat (silu_out_w),
        .sp_out_flat   (sp_out_w),
        .exp_out_flat  (exp_out_w),
        .rsqrt_idx     (rsqrt_idx_r),
        .rsqrt_data    (rsqrt_data)
    );

    // ------------------------------------------------------------------------
    // MAC chain (S_MAC1..5): y[g] = W[g] . x over len inputs, 16 outputs per group, two weights per cycle (MAC2).
    // A stage on the chain is one row of the job table below. Mamba-phase jobs step ctr_g by 2 from CLUSTER_ID
    // (weights compacted per cluster); time-split jobs (Inception phase) run all groups of t = CLUSTER_ID, +2, ...
    // with full weights, their input / output bases stepped per t in tin_r / tout_r.
    // ------------------------------------------------------------------------
    function [14:0] mac_w_base_of;
        input [4:0] stg;
        case (stg)
            STG_M1A: mac_w_base_of = W_INPROJ_BASE;
            STG_P1:  mac_w_base_of = W_P1_BASE;
            STG_B1:  mac_w_base_of = W_B1_BASE;
            STG_BOT: mac_w_base_of = W_BOT_BASE;
            STG_B2:  mac_w_base_of = W_B2_BASE;
            STG_B3:  mac_w_base_of = W_B3_BASE;
            STG_B4:  mac_w_base_of = W_B4_BASE;
            STG_ENC: mac_w_base_of = W_ENC_BASE;
            STG_FC:  mac_w_base_of = W_FC_BASE;
            default: mac_w_base_of = W_OUTPROJ_BASE;
        endcase
    endfunction

    function mac_tsplit_of;
        input [4:0] stg;
        mac_tsplit_of = (stg == STG_P1) || (stg == STG_MP3) || (stg == STG_B1) || (stg == STG_BOT) ||
                        (stg == STG_B2) || (stg == STG_B3) || (stg == STG_B4) || (stg == STG_ENC);
    endfunction

    // Time-split jobs: bases at t = 0 and words per t. Branch b writes words b*CH_OUT/4.. of each t of INC_OUT.
    function [14:0] tsplit_in_base_of;
        input [4:0] stg;
        case (stg)
            STG_MP3, STG_BOT:       tsplit_in_base_of = PT_INPUT;
            STG_B1:                 tsplit_in_base_of = PT_MP3;
            STG_B2, STG_B3, STG_B4: tsplit_in_base_of = PT_BOT;
            STG_ENC:                tsplit_in_base_of = PT_ENC_IN;
            default:                tsplit_in_base_of = PT_X_IN;
        endcase
    endfunction
    function [14:0] tsplit_out_base_of;
        input [4:0] stg; input [8:0] ch_out;
        case (stg)
            STG_MP3: tsplit_out_base_of = PT_MP3;
            STG_B1:  tsplit_out_base_of = PT_INC_OUT;
            STG_BOT: tsplit_out_base_of = PT_BOT;
            STG_ENC: tsplit_out_base_of = PT_X_IN;
            STG_B2:  tsplit_out_base_of = PT_INC_OUT + {8'b0, ch_out[8:2]};
            STG_B3:  tsplit_out_base_of = PT_INC_OUT + {7'b0, ch_out[8:2], 1'b0};
            STG_B4:  tsplit_out_base_of = PT_INC_OUT + {7'b0, ch_out[8:2], 1'b0} + {8'b0, ch_out[8:2]};
            default: tsplit_out_base_of = PT_INPUT;
        endcase
    endfunction
    function [8:0] tsplit_in_wpt_of;
        input [4:0] stg; input [8:0] ch_in; input [8:0] ch_out;
        tsplit_in_wpt_of = (stg == STG_ENC) ? 9'd1 :
                           (stg == STG_P1)  ? ch_in :
                           (stg == STG_B2 || stg == STG_B3 || stg == STG_B4) ? {2'b0, ch_out[8:2]} : ch_out;
    endfunction
    function [8:0] tsplit_out_wpt_of;
        input [4:0] stg; input [8:0] ch_in; input [8:0] ch_out;
        tsplit_out_wpt_of = (stg == STG_BOT) ? {2'b0, ch_out[8:2]} :
                            (stg == STG_ENC) ? ch_in : ch_out;
    endfunction

    // Split-K: in the MAC jobs the PE array takes input words [0, wp), wp = ceil(w/2), and the scan rows (aux
    // engine) take [wp, w) from weight bank X.
    function sk_of;
        input [4:0] stg;
        sk_of = (stg == STG_M1A) || (stg == STG_M8) || (stg == STG_FC) || (mac_tsplit_of(stg) && stg != STG_MP3);
    endfunction

    // job_r follows job_of(cur_stage) one cycle late and is preloaded with the next stage's row at the barrier
    // release, which keeps the stage decode (and the conv K*dim multiply) off the FSM paths. len is the PE's
    // share of the input elements; lenx the aux engine's, wp the PE's word count.
    localparam JOB_W = 141;
    function [JOB_W-1:0] job_of;
        input [4:0] stg;
        reg [13:0] len; reg [9:0] grp; reg [14:0] inb, wrb; reg [4:0] nstg; reg [6:0] nst;
        reg ben; reg [14:0] bbase; reg conv; reg [14:0] pad; reg tsp; reg [8:0] iwpt, owpt;
        reg [9:0] wds, wp; reg [13:0] lenx;
        begin
            ben = 1'b0; bbase = 15'd0; conv = 1'b0; pad = 15'd0;
            tsp  = mac_tsplit_of(stg);
            iwpt = tsplit_in_wpt_of (stg, CH_IN, CH_OUT);
            owpt = tsplit_out_wpt_of(stg, CH_IN, CH_OUT);
            case (stg)
                STG_ENC: begin                     // 12 leads padded to one word, + bias
                    len = 14'd16;                 grp = {1'b0, CH_IN};
                    inb = PT_ENC_IN;              wrb = PT_X_IN;
                    ben = 1'b1;                   bbase = C_ENC_BIAS_BASE;
                    nstg = STG_P1;                nst = S_MAC1;
                end
                STG_P1: begin                      // k1 conv + fused BN bias
                    len = {1'b0, CH_IN, 4'b0};    grp = {1'b0, CH_OUT};
                    inb = PT_X_IN;                wrb = PT_INPUT;
                    ben = 1'b1;                   bbase = C_P1_BIAS_BASE;
                    nstg = STG_MP3;               nst = S_MP3;
                end
                STG_MP3: begin                     // not a MAC job; S_MP3 uses grp only
                    len = {1'b0, CH_OUT, 4'b0};   grp = {1'b0, CH_OUT};
                    inb = PT_INPUT;               wrb = PT_MP3;
                    nstg = STG_B1;                nst = S_MAC1;
                end
                STG_B1: begin
                    len = {1'b0, CH_OUT, 4'b0};   grp = {3'b0, CH_OUT[8:2]};
                    inb = PT_MP3;                 wrb = PT_INC_OUT;
                    nstg = STG_BOT;               nst = S_MAC1;
                end
                STG_BOT: begin
                    len = {1'b0, CH_OUT, 4'b0};   grp = {3'b0, CH_OUT[8:2]};
                    inb = PT_INPUT;               wrb = PT_BOT;
                    nstg = STG_B2;                nst = S_MAC1;
                end
                // K-tap convs over BOT (dim = CH_OUT*4 channels): one accumulation over k = tap*dim + ci,
                // input words before t-pad or past T read as zero.
                STG_B2: begin
                    len = {CH_OUT, 2'b0} * 14'd9; grp = {3'b0, CH_OUT[8:2]};
                    inb = PT_BOT;                 wrb = PT_INC_OUT;
                    pad = {CH_OUT, 2'b0} * 15'd4 / 15'd16;   conv = 1'b1;
                    nstg = STG_B3;                nst = S_MAC1;
                end
                STG_B3: begin
                    len = {CH_OUT, 2'b0} * 14'd19; grp = {3'b0, CH_OUT[8:2]};
                    inb = PT_BOT;                  wrb = PT_INC_OUT;
                    pad = {CH_OUT, 2'b0} * 15'd9 / 15'd16;   conv = 1'b1;
                    nstg = STG_B4;                 nst = S_MAC1;
                end
                STG_B4: begin
                    len = {CH_OUT, 2'b0} * 14'd39; grp = {3'b0, CH_OUT[8:2]};
                    inb = PT_BOT;                  wrb = PT_INC_OUT;
                    pad = {CH_OUT, 2'b0} * 15'd19 / 15'd16;  conv = 1'b1;
                    nstg = stg;                    nst = S_DONE;
                end
                STG_M1A: begin                     // in_proj into frame t mod 4
                    len = {1'b0, d_model};        grp = inp_g;
                    inb = t_cnt[0] ? PT_X_NORM_B : PT_X_NORM;
                    wrb = fb_r[t_cnt[1:0]];
                    nstg = STG_M2;                nst = S_CONV0;
                end
                STG_M8: begin                      // out_proj
                    len = d_inner;                grp = {1'b0, CH_OUT};
                    inb = PT_Y_NORM;              wrb = PT_MAMBA_OUT + t_ch_r;
                    nstg = (t_cnt != t_last) ? STG_RN : FIN_EN ? STG_FIN : GAP_EN ? STG_GAP : stg;
                    nst  = (t_cnt != t_last) ? S_RN1  : FIN_EN ? S_FIN   : GAP_EN ? S_GAP   : S_DONE;
                end
                STG_FC: begin                      // one output group; cluster 1's copy lands in PT_LOGITS+1
                    len = {1'b0, CH_OUT, 4'b0};   grp = 10'd1;
                    inb = PT_GAP;                 wrb = PT_LOGITS;
                    ben = 1'b1;                   bbase = C_FC_BIAS_BASE;
                    nstg = stg;                   nst = S_DONE;
                end
                default: begin
                    len = {1'b0, d_model};        grp = ch_m_act;
                    inb = PT_X_NORM;              wrb = PT_MAMBA_OUT + t_ch_r;
                    nstg = stg;                   nst = S_IDLE;
                end
            endcase
            wds  = len[13:4];
            wp   = (sk_of(stg) && wds >= 10'd2) ? (wds + 10'd1) >> 1 : wds;
            lenx = len - {wp, 4'b0};
            len  = {wp, 4'b0};
            job_of = {lenx, wp, len, grp, inb, wrb, nstg, nst, ben, bbase, conv, pad, tsp, iwpt, owpt};
        end
    endfunction

    // = (job_of(stg) has lenx != 0): split jobs of at least two words (the encoder has one, P1 has CH_IN)
    function ax_on_of;
        input [4:0] stg;
        ax_on_of = sk_of(stg) && (stg != STG_ENC) && (stg != STG_P1 || CH_IN[8:1] != 8'd0);
    endfunction

    reg  [JOB_W-1:0] job_r;
    reg  [14:0]      mac_w_base;
    (* max_fanout = 32 *) reg ax_on_r;     // the current job has an aux share
    wire             bar_rel = (state == S_BARRIER) && bar_go;
    wire [JOB_W-1:0] job_c   = bar_rel ? job_of(next_stage_q) : job_of(cur_stage);
    wire             ax_on_n = ax_on_of(bar_rel ? next_stage_q : cur_stage);
    always @(posedge clk or posedge rst)
        if (rst) begin
            job_r      <= {JOB_W{1'b0}};
            mac_w_base <= 15'd0;
            ax_on_r    <= 1'b0;
        end else begin
            job_r      <= job_c;
            mac_w_base <= bar_rel ? mac_w_base_of(next_stage_q) : mac_w_base_of(cur_stage);
            ax_on_r    <= ax_on_n;
        end
    wire [13:0] mac_len_x      = job_r[140:127];
    wire [9:0]  mac_wp         = job_r[126:117];
    wire [13:0] mac_len        = job_r[116:103];
    wire [9:0]  mac_grp_count  = job_r[102:93];
    wire [14:0] mac_in_base    = job_r[92:78];
    wire [14:0] mac_wr_base    = job_r[77:63];
    wire [4:0]  mac_next_stage = job_r[62:58];
    wire [6:0]  mac_next_state = job_r[57:51];
    wire        mac_bias_en    = job_r[50];
    wire [14:0] mac_bias_base  = job_r[49:35];
    wire        mac_conv       = job_r[34];
    wire [14:0] mac_pad_words  = job_r[33:19];
    wire        mac_tsplit     = job_r[18];
    wire [8:0]  mac_in_wpt     = job_r[17:9];
    wire [8:0]  mac_out_wpt    = job_r[8:0];

    reg  [14:0] tin_r, tout_r;
    wire [14:0] tin_next  = tin_r  + {5'b0, mac_in_wpt,  1'b0};   // t + 2
    wire [14:0] tout_next = tout_r + {5'b0, mac_out_wpt, 1'b0};
    wire [14:0] mac_in_base_eff = mac_tsplit ? tin_r - mac_pad_words : mac_in_base;
    wire [14:0] mac_wr_addr     = (mac_tsplit ? tout_r : mac_wr_base) + {6'b0, ctr_g};
    wire        mac_g_last_in_t = ({1'b0, ctr_g} + 10'd1 >= mac_grp_count);
    wire        mac_t_last      = ({1'b0, t_cnt} + 11'd2 >= {1'b0, T_MAX});

    wire [12:0] ctr_k_p1 = ctr_k + 13'd1;
    wire [12:0] ctr_k_p2 = ctr_k + 13'd2;
    wire [12:0] ctr_k_p3 = ctr_k + 13'd3;
    wire [12:0] ctr_k_p4 = ctr_k + 13'd4;
    wire [12:0] ctr_k_p6 = ctr_k + 13'd6;

    // x operands of the MAC2: elements ctr_k and ctr_k+1 of the current input word, broadcast to all lanes
    wire [15:0] y_scalar  = m_rd_data[ctr_k[3:0]*`DATA_W +: `DATA_W];
    wire [3:0]  y_lane2   = ctr_k[3:0] + 4'd1;
    wire [15:0] y_scalar2 = m_rd_data[y_lane2*`DATA_W +: `DATA_W];
    (* keep = "true" *) wire [`NUM_LANES*`DATA_W-1:0] y_broadcast  = {`NUM_LANES{y_scalar}};
    (* keep = "true" *) wire [`NUM_LANES*`DATA_W-1:0] y_broadcast2 = {`NUM_LANES{y_scalar2}};

    // Conv zero padding: the input word ctr_k>>4 is padding outside [in_base, region_end). The limits move only
    // with tin_r or the stage, at least 2 cycles before the next MAC, so they are registered and the per-cycle
    // test is one compare of the word index.
    wire [14:0] conv_region_end = mac_in_base + ((mac_in_wpt == 9'd2) ? {4'b0, T_MAX, 1'b0} : {5'b0, T_MAX});
    wire signed [16:0] conv_base_s = mac_tsplit ? $signed({2'b0, tin_r}) - $signed({2'b0, mac_pad_words})
                                                : $signed({2'b0, mac_in_base});
    reg  signed [16:0] conv_lo_r, conv_hi_r;
    always @(posedge clk or posedge rst)
        if (rst) begin
            conv_lo_r <= 17'sd0;
            conv_hi_r <= 17'sd0;
        end else begin
            conv_lo_r <= $signed({2'b0, mac_in_base})     - conv_base_s;
            conv_hi_r <= $signed({2'b0, conv_region_end}) - conv_base_s;
        end
    wire signed [16:0] conv_kw = $signed({8'b0, ctr_k[12:4]});
    wire conv_zero = mac_conv && ((conv_kw < conv_lo_r) || (conv_kw >= conv_hi_r));
    wire [`NUM_LANES*`DATA_W-1:0] mac_y1 = conv_zero ? {`NUM_LANES*`DATA_W{1'b0}} : y_broadcast;
    wire [`NUM_LANES*`DATA_W-1:0] mac_y2 = conv_zero ? {`NUM_LANES*`DATA_W{1'b0}} : y_broadcast2;

    // Reads run 6 elements ahead of the MAC: the word must be on m_rd_data when ctr_k[3:0] wraps.
    wire [14:0] mac_rd_addr_k0 = mac_in_base_eff + {6'b0, ctr_k[12:4]};
    wire [14:0] mac_rd_addr_k2 = mac_in_base_eff + {6'b0, ctr_k_p2[12:4]};
    wire [14:0] mac_rd_addr_k6 = mac_in_base_eff + {6'b0, ctr_k_p6[12:4]};

    // Weight base of the current group: loaded at the barrier release and at the group loopback, held inside the
    // loop. The next group's base is a registered multiply, first used >= 16 cycles after ctr_g changes.
    reg  [14:0] w_grp_base_r, w_grp_base_next_r;
    reg  [8:0]  w_grp_idx_next_r;
    wire [8:0]  ctr_g_p2   = ctr_g + 9'd2;
    wire [8:0]  g_bank_idx = {1'b0, ctr_g[8:1]};                 // group index inside this cluster's weights
    wire [8:0]  w_grp_idx_next = mac_tsplit ? (mac_g_last_in_t ? 9'd0 : ctr_g + 9'd1) : {1'b0, ctr_g_p2[8:1]};
    wire [14:0] mac_w_addr_k0 = w_grp_base_r + {2'b0, ctr_k};
    wire [14:0] mac_w_addr_k2 = w_grp_base_r + {2'b0, ctr_k_p2};
    wire [14:0] mac_w_addr_k4 = w_grp_base_r + {2'b0, ctr_k_p4};
    wire [14:0] mac_w_addr_k6 = w_grp_base_r + {2'b0, ctr_k_p6};
    wire [14:0] mac_rd_addr_g0 = (mac_tsplit && mac_g_last_in_t) ? tin_next - mac_pad_words : mac_in_base_eff;

    // Registered: ctr_g and t_cnt change only at a group's end, and the flag is next read at the next group's end.
    reg  mac_last_iter;
    // = (ctr_k == mac_len) in S_MAC4, where ctr_k steps by 2 from 2 and mac_len >= 16
    reg  mac_k_end;
    always @(posedge clk or posedge rst)
        if (rst) begin
            mac_last_iter <= 1'b0;
            mac_k_end     <= 1'b0;
        end else begin
            mac_last_iter <= mac_tsplit ? (mac_g_last_in_t && mac_t_last)
                                        : (ctr_g_p2 >= {1'b0, mac_grp_count[8:0]});
            mac_k_end     <= (state == S_MAC4) && ({1'b0, ctr_k_p2} == mac_len);
        end
    // Drain overlap: every group but the last leaves S_MAC4 without draining; the next group starts at once and
    // this group's result is written 6 cycles later from S_MAC4 (see ovl_*).
    wire mac_ovl_exit = (state == S_MAC4) && mac_k_end && !mac_last_iter;
    reg                           ovl_pend;
    reg  [2:0]                    ovl_cnt;
    reg  [14:0]                   ovl_addr;
    reg  [`NUM_LANES*`DATA_W-1:0] ovl_data;
    reg  [`NUM_LANES*`DATA_W-1:0] mac_bsum_r;    // result + bias, registered (keeps the add off m_wr_data)

    always @(posedge clk or posedge rst)
        if (rst) begin
            w_grp_idx_next_r  <= 9'd0;
            w_grp_base_next_r <= 15'd0;
            w_grp_base_r      <= 15'd0;
            t_ch_r            <= 15'd0;
        end else begin
            w_grp_idx_next_r  <= w_grp_idx_next;
            w_grp_base_next_r <= mac_w_base + ({6'b0, w_grp_idx_next_r} * {1'b0, mac_len});
            if ((state == S_IDLE || state == S_DONE) && start)
                t_ch_r <= 15'd0;
            if (bar_rel) begin
                w_grp_base_r <= mac_w_base_of(next_stage_q);
                if (cur_stage == STG_M8 && next_stage_q == STG_RN)
                    t_ch_r <= t_ch_r + {6'b0, CH_OUT};
            end
            if ((state == S_MAC5 && wait_cnt == 3'd4 && !mac_last_iter) || mac_ovl_exit)
                w_grp_base_r <= w_grp_base_next_r;
        end

    // ------------------------------------------------------------------------
    // Split-K aux engine: the scan's C*h row (tree A, rows 0-7) and gate row (tree B, rows 8-15) reduce one weight
    // row of bank X against the aux x word per cycle. Bank X word = base + g*lenx + jj*16 + r, lane = input. With
    // PE cycle m (ctr_k = 2m, m = 0 at S_MAC3), aux row n = 8jj + c has its operands at m = n + 1:
    //   m = n - 3   bank X read of row n (port B), m = n - 2 of row n + 8 (port A)
    //   m = 8jj - 2 main RAM read of x word wp + jj, on a slot whose PE read repeats the one before (S_MAC2 wait 0,
    //               or ctr_k[3:0] = 12); captured into h_new_r at m = 8jj while m_rd_data holds
    //   m = n + 4   ch_b / gt_b reach the rotating accumulators (8 entries each, rows in order after a word)
    // For w even the last row lands with the PE's last acc_raw (m = 8wp + 4); for w odd one word earlier. The
    // accumulators then hold until the next group's first row at m' = 5, so comb_out_r is sampled where out_val was.
    // ------------------------------------------------------------------------
    reg  [14:0] wx_grp_base_r, wx_grp_base_next_r;
    reg  [8:0]  ax_ra;                  // next bank X row to read
    reg         ax_slot_r, ax_sd1, ax_pad_r;
    reg  [14:0] ax_rd_addr_r;
    reg         ax_t1, ax_t2, ax_t3, ax_f1, ax_f2, ax_f3;
    wire [9:0]  ax_words  = mac_len_x[13:4];
    wire        ax_live   = ax_on_r && (state == S_MAC4) && (ctr_k <= mac_len_x[12:0]);
    wire signed [16:0] ax_wa = $signed({7'b0, mac_wp}) + $signed({8'b0, ctr_k_p2[12:4]});
    always @(posedge clk or posedge rst)
        if (rst) begin
            wx_grp_base_r      <= 15'd0;
            wx_grp_base_next_r <= 15'd0;
            wx_rd_addr         <= 15'd0;
            wx_rd_addr2        <= 15'd0;
            ax_ra              <= 9'd0;
            ax_slot_r          <= 1'b0;
            ax_sd1             <= 1'b0;
            ax_cap             <= 1'b0;
            ax_pad_r           <= 1'b0;
            ax_rd_addr_r       <= 15'd0;
            ax_t1 <= 1'b0; ax_t2 <= 1'b0; ax_t3 <= 1'b0;
            ax_f1 <= 1'b0; ax_f2 <= 1'b0; ax_f3 <= 1'b0;
        end else begin
            wx_grp_base_next_r <= (w_grp_idx_next_r == 9'd0) ? mac_w_base : wx_grp_base_r + {1'b0, mac_len_x};
            if (bar_rel)
                wx_grp_base_r <= mac_w_base_of(next_stage_q);
            if ((state == S_MAC5 && wait_cnt == 3'd4 && !mac_last_iter) || mac_ovl_exit)
                wx_grp_base_r <= wx_grp_base_next_r;

            wx_rd_addr2 <= wx_rd_addr | 15'd8;
            if (state == S_MAC1 || mac_ovl_exit) begin
                wx_rd_addr <= (state == S_MAC1) ? wx_grp_base_r : wx_grp_base_next_r;
                ax_ra      <= 9'd1;
            end else if ({4'b0, ax_ra} < mac_len_x[13:1]) begin
                wx_rd_addr <= wx_grp_base_r + {5'b0, ax_ra[8:3], 1'b0, ax_ra[2:0]};
                ax_ra      <= ax_ra + 9'd1;
            end

            ax_slot_r    <= ax_on_r && (state == S_MAC1 || mac_ovl_exit ||
                            (state == S_MAC4 && ctr_k[3:0] == 4'd10 && {1'b0, ctr_k[12:4]} + 10'd1 < ax_words));
            ax_rd_addr_r <= (mac_ovl_exit ? mac_rd_addr_g0
                                          : mac_in_base_eff + ((state == S_MAC4) ? {6'b0, ctr_k[12:4]} + 15'd1 : 15'd0))
                            + {5'b0, mac_wp};
            ax_sd1       <= ax_slot_r;
            ax_cap       <= ax_sd1;
            ax_pad_r     <= mac_conv && ((ax_wa < conv_lo_r) || (ax_wa >= conv_hi_r));

            ax_t1 <= ax_live;  ax_t2 <= ax_t1;  ax_t3 <= ax_t2;
            ax_f1 <= (ctr_k <= 13'd16);  ax_f2 <= ax_f1;  ax_f3 <= ax_f2;
        end

    // ------------------------------------------------------------------------
    // MP3: max(x[t-1], x[t], x[t+1]) with the edges clamped
    // ------------------------------------------------------------------------
    reg  [1:0]  mp_ph;
    reg         mp_fv, mp_drain;    // a fold is due (reads issued 3 cycles ago); last item's reads issued
    reg  [`NUM_LANES*`DATA_W-1:0] mp_acc;
    reg                           mp_wr;
    reg  [14:0]                   mp_wr_addr, mp_wa_pend;
    // (g, t) step at phase 2; t-1 and t+1 are read at phases 1 and 2, so their addresses are registered.
    wire [14:0] mp_cur_addr  = tin_r + {6'b0, ctr_g};
    reg  [14:0] mp_prev_addr, mp_next_addr;
    always @(posedge clk or posedge rst)
        if (rst) begin
            mp_prev_addr <= 15'd0;
            mp_next_addr <= 15'd0;
        end else begin
            mp_prev_addr <= (t_cnt == 10'd0) ? mp_cur_addr : mp_cur_addr - {6'b0, CH_OUT};
            mp_next_addr <= ({1'b0, t_cnt} + 11'd1 >= {1'b0, T_MAX}) ? mp_cur_addr : mp_cur_addr + {6'b0, CH_OUT};
        end

    // ------------------------------------------------------------------------
    // RN: sum of squares over all of x[t] (both clusters, same result), rsqrt, then the pipelined apply
    // x_norm = sat(sat(x*g >> 11) * S >> 11) for this cluster's groups, or for all groups when hoisted.
    // ------------------------------------------------------------------------
    reg         rn_hoist;           // cluster 1 computes RN of t+1 at the end of its in_proj of t
    reg  [15:0] rn_scale;
    wire [14:0] rn_t_off  = t_ch_r + (rn_hoist ? {6'b0, CH_OUT} : 15'd0);
    wire [14:0] rn_xn_base = (t_cnt[0] ^ rn_hoist) ? PT_X_NORM_B : PT_X_NORM;
    wire [14:0] rn_sq_addr_k0 = PT_INPUT + rn_t_off + {8'b0, ctr_k[6:0]};
    wire [14:0] rn_sq_addr_k1 = PT_INPUT + rn_t_off + {8'b0, ctr_k_p1[6:0]};
    wire [14:0] rn_sq_addr_k2 = PT_INPUT + rn_t_off + {8'b0, ctr_k_p2[6:0]};
    wire [14:0] rn_sq_addr_k3 = PT_INPUT + rn_t_off + {8'b0, ctr_k_p3[6:0]};
    function [12:0] rn_clip;                // rsqrt index of a sum of squares
        input signed [`ACC_W+4:0] m;
        rn_clip = (m < 0) ? 13'd0 : (m > 45'sd8191) ? 13'd8191 : m[12:0];
    endfunction
    // Apply slots for group j (counter ps): 2j read x and g, 2j+3 MUL x*g, 2j+9 latch, 2j+10 MUL by S, 2j+16 write
    wire [8:0]  rn_n      = rn_hoist ? CH_OUT : {1'b0, CH_OUT[8:1]};
    wire [8:0]  rn_pm3    = ps - 9'd3;
    wire [8:0]  rn_pm9    = ps - 9'd9;
    wire [8:0]  rn_pm10   = ps - 9'd10;
    wire [8:0]  rn_pm16   = ps - 9'd16;
    wire        rn_do_rd  = !ps[0]                && ({1'b0, ps[8:1]}      < rn_n);
    wire        rn_do_m1  =  ps[0] && ps >= 9'd3  && ({1'b0, rn_pm3[8:1]}  < rn_n);
    wire        rn_do_l   =  ps[0] && ps >= 9'd9  && ({1'b0, rn_pm9[8:1]}  < rn_n);
    wire        rn_do_m2  = !ps[0] && ps >= 9'd10 && ({1'b0, rn_pm10[8:1]} < rn_n);
    wire        rn_do_wr  = !ps[0] && ps >= 9'd16 && ({1'b0, rn_pm16[8:1]} < rn_n);
    wire [8:0]  rn_grp_rd = rn_hoist ? {1'b0, ps[8:1]}      : {ps[8:1], CLUSTER_ID[0]};
    wire [8:0]  rn_grp_wr = rn_hoist ? {1'b0, rn_pm16[8:1]} : {rn_pm16[8:1], CLUSTER_ID[0]};
    wire [8:0]  rn_ps_end = {rn_n[7:0], 1'b0} + 9'd14;

    // ------------------------------------------------------------------------
    // M2 conv: tap stream q = 4j + tap over this cluster's n = xbc_g/2 groups (global group 2j + CLUSTER_ID).
    //   q      read x of frame (t-3+tap) mod 4 and the tap weight
    //   q+3    MAC (clear on tap 0); zero input when t-3+tap < 0
    //   4j+9   read the bias
    //   4j+12  m2_res = sat(sum + bias)       (the tap-3 MAC fired at 4j+6)
    //   4j+13  write m2_res to XCONV          (read back by the testbench only)
    //   4j+14  write SiLU(m2_res) to XBC
    // ------------------------------------------------------------------------
    wire [8:0]  m2_n      = {1'b0, xbc_g[8:1]};
    wire [8:0]  m2_pm3    = ps - 9'd3;
    wire [8:0]  m2_pm9    = ps - 9'd9;
    wire [8:0]  m2_pm12   = ps - 9'd12;
    wire [8:0]  m2_pm13   = ps - 9'd13;
    wire [8:0]  m2_pm14   = ps - 9'd14;
    wire [1:0]  m2_tapf   = ps[1:0];
    wire [1:0]  m2_slotf  = t_cnt[1:0] + 2'd1 + m2_tapf;         // (t - 3 + tap) mod 4
    wire [1:0]  m2_tapm   = m2_pm3[1:0];
    wire        m2_pad    = ({1'b0, t_cnt} + {9'b0, m2_tapm}) < 11'd3;
    wire        m2_do_f   =                                         ({2'b0, ps[8:2]}      < m2_n);
    wire        m2_do_mac = ps >= 9'd3                           && ({2'b0, m2_pm3[8:2]}  < m2_n);
    wire        m2_do_b   = ps >= 9'd9  && m2_pm9[1:0]  == 2'd0  && ({2'b0, m2_pm9[8:2]}  < m2_n);
    wire        m2_do_r   = ps >= 9'd12 && m2_pm12[1:0] == 2'd0  && ({2'b0, m2_pm12[8:2]} < m2_n);
    wire        m2_do_w   = ps >= 9'd13 && m2_pm13[1:0] == 2'd0  && ({2'b0, m2_pm13[8:2]} < m2_n);
    wire        m2_do_s   = ps >= 9'd14 && m2_pm14[1:0] == 2'd0  && ({2'b0, m2_pm14[8:2]} < m2_n);
    wire [8:0]  m2_grp_f  = {1'b0, ps[8:2],      CLUSTER_ID[0]};
    wire [8:0]  m2_grp_b  = {1'b0, m2_pm9[8:2],  CLUSTER_ID[0]};
    wire [8:0]  m2_grp_w  = {1'b0, m2_pm13[8:2], CLUSTER_ID[0]};
    wire [8:0]  m2_grp_s  = {1'b0, m2_pm14[8:2], CLUSTER_ID[0]};
    wire [8:0]  m2_ps_end = {m2_n[6:0], 2'b0} + 9'd10;

    // ------------------------------------------------------------------------
    // M6 scan. Per-head values (dt, dA, D) are lanes of one word, lane = head = global group >> PSH; per-channel
    // values (x, x*dt, y) are lanes of the group's word. Slot i = (channel c, state word s), i = c*S_CT + s.
    // The PE issues one MUL2 per slot (dA*h, xdt*B); C*h_new (+ D*x on s = 0) runs on a fabric multiplier row and
    // adder tree, one exact sum per channel and one floor shift. A group lasts GP = NS + 2 cycles; its tail
    // runs on under the next group, carried by a token delay line (one token per slot, pushed with its MUL2).
    // ------------------------------------------------------------------------
    localparam [3:0] S_CT = `M2_S;
    localparam       SL   = (S_CT == 4'd8) ? 3 : (S_CT == 4'd4) ? 2 : (S_CT == 4'd2) ? 1 : 0;
    localparam       PSH  = `M2_PSH;
    localparam [8:0] NS   = 9'd16 * S_CT;
    localparam [8:0] GP   = NS + 9'd2;
    localparam       TK   = 14;
    localparam [4:0] LD_END = ({S_CT, 1'b0} + 5'd4 > 5'd11) ? {S_CT, 1'b0} + 5'd4 : 5'd11;

    reg  [4:0]                    ld;
    reg  [`NUM_LANES*`DATA_W-1:0] dt_word, dA_word, D_word;     // lane = head
    reg  [`NUM_LANES*`DATA_W-1:0] x_word, xdt_word;             // lane = channel
    reg  [`NUM_LANES*`DATA_W-1:0] x_next, xdt_next;             // next group's, prefetched under this group
    reg  [`NUM_LANES*`DATA_W-1:0] silu_z_word, y_ssm_word;
    reg                           scan_more_r;    // ctr_g_p2 < ch_m_act
    always @(posedge clk or posedge rst)
        if (rst) scan_more_r <= 1'b0;
        else     scan_more_r <= ctr_g_p2 < {1'b0, ch_m_act[8:0]};
    reg  [`NUM_LANES*`DATA_W-1:0] B_w [0:S_CT-1];              // one word per state word, per timestep
    reg  [`NUM_LANES*`DATA_W-1:0] C_w [0:S_CT-1];
    reg  [`NUM_LANES*`DATA_W-1:0] h_new_r;                      // also a scratch latch in RN and GN
    reg  signed [47:0]            y_acc;
    wire [14:0] dt_raw_addr = fbx_r[t_cnt[1:0]] + {5'b0, xbc_g};

    wire [`NUM_LANES*`DATA_W-1:0] h_new_w;
    genvar hl;
    generate
        for (hl = 0; hl < `NUM_LANES; hl = hl + 1) begin : H_NEW
            assign h_new_w[hl*`DATA_W +: `DATA_W] = sat_add16(cl_out_vec[hl*`DATA_W +: `DATA_W],
                                                              cl_out_vec2[hl*`DATA_W +: `DATA_W]);
        end
    endgenerate

    wire [8:0]  head9    = ctr_g >> PSH;
    wire [3:0]  head     = head9[3:0];
    wire [8:0]  head9_n  = ctr_g_p2 >> PSH;
    wire [3:0]  head_n   = head9_n[3:0];
    wire [15:0] dt_s     = dt_word[{head, 4'b0} +: 16];
    wire [15:0] dt_s_n   = dt_word[{head_n, 4'b0} +: 16];
    wire [15:0] dA_s     = dA_word[{head, 4'b0} +: 16];
    wire [15:0] D_s      = D_word [{head, 4'b0} +: 16];

    wire [8:0]  sl_pm2   = ps - 9'd2;
    wire [7:0]  sl_i     = sl_pm2[7:0];                  // slot of the MUL2 issued now
    wire [7:0]  sl_c_w   = sl_i >> SL;
    wire [3:0]  sl_c     = sl_c_w[3:0];
    wire [7:0]  sl_s_w   = sl_i & {4'b0, S_CT - 4'd1};
    wire [2:0]  sl_s     = sl_s_w[2:0];
    wire        sl_issue = (ps >= 9'd2) && (ps <= NS + 9'd1);
    wire [15:0] h_base   = {7'b0, g_bank_idx} << (4 + SL);   // H address of slot 0 of this group
    wire [15:0] xdt_c    = xdt_word[{sl_c, 4'b0} +: 16];
    wire [15:0] x_c      = x_word  [{sl_c, 4'b0} +: 16];

    // Token of a slot; tk_*[k] is visible k+1 cycles after its MUL2 is assigned (slot i: ps = i + 3 + k)
    reg              tk_v  [0:TK-1];
    reg              tk_sf [0:TK-1];                     // first state word of the channel
    reg              tk_sl [0:TK-1];                     // last state word of the channel
    reg              tk_lg [0:TK-1];                     // last slot of the group
    reg  [2:0]       tk_s  [0:TK-1];
    reg  [3:0]       tk_c  [0:TK-1];
    reg  [H_ADDR_W-1:0] tk_wa [0:TK-1];
    reg  [15:0]      tk_x  [0:TK-1];
    reg  [15:0]      tk_D  [0:TK-1];
    reg  [8:0]       tk_g  [0:TK-1];

    // Fabric rows. Gate row: y_gated = sat((y_ssm * SiLU(z)) >> 11), shared with the next group's x*dt (the PE has
    // no free slot); product, then shift + saturate. C*h row: 16 products, a 3-level adder tree, plus D*x.
    // The gate row also squares words into tree B (exact sums): each y_gated word for GN (gn_acc), and on cluster 1
    // one word of x[t+2] per group for the hoisted RN of t+2 (rn_acc; groups per cluster = CH_OUT). Gate-row slots
    // of a group: ps 4 x*dt, 8 x[t+2] square, 11 previous group's gate, 13 its y_gated square.
    reg  signed [31:0]            gt_p [0:`NUM_LANES-1];
    reg  [`NUM_LANES*`DATA_W-1:0] gt_out;
    reg                           gt_go_g, gt_go_x, gt_go_gd, gt_go_xd;
    reg  [1:0]                    gt_sel;    // registered operand select: 0 aux, 1 gate, 2 square, 3 x*dt
    reg  [`NUM_LANES*`DATA_W-1:0] sq_op_r;   // the word being squared
    reg                           sq_g, sq_r;             // square issued: y_gated (GN), x[t+2] (RN)
    reg  [3:1]                    sq_g_d;
    reg  [5:1]                    sq_r_d;
    reg                           rn_ld_r, rn_pre_on_r;
    reg  [14:0]                   rn_pre_addr_r;
    reg  signed [44:0]            gn_acc, rn_acc;
    reg  [15:0]                   gt_dt;     // registered: a head mux straight into the DSPs failed timing
    wire [`NUM_LANES*`DATA_W-1:0] gt_sat;
    reg  signed [31:0] ch_p [0:`NUM_LANES-1];
    reg  [`NUM_LANES*`DATA_W-1:0] ch_b_op_r;   // C word (scan) or bank X row (aux)
    // Tree B stays in fabric: as a DSP cascade Vivado duplicated every odd gate product (gt_p also feeds gt_sat)
    (* use_dsp = "no" *) reg signed [32:0] gt_a [0:7];
    (* use_dsp = "no" *) reg signed [34:0] gt_b [0:1];
    reg  signed [`ACC_W-1:0] ax_acc_a [0:7];
    reg  signed [`ACC_W-1:0] ax_acc_b [0:7];
    reg  [`NUM_LANES*`DATA_W-1:0] comb_out_r;  // MAC group result: sat16((acc_raw + aux) >>> 11)
    reg  signed [31:0] dx_p, dx_a, dx_b;
    reg  signed [32:0] ch_a [0:7];
    reg  signed [34:0] ch_b [0:1];

    wire signed [47:0] y_shr = y_acc >>> `FRAC_BITS;
    wire signed [15:0] y_sat = (y_shr > 48'sd32767) ? 16'sh7FFF : (y_shr < -48'sd32768) ? 16'sh8000 : y_shr[15:0];

    genvar gl;
    generate
        for (gl = 0; gl < `NUM_LANES; gl = gl + 1) begin : GT_SAT
            assign gt_sat[gl*`DATA_W +: `DATA_W] = ((gt_p[gl] >>> `FRAC_BITS) > 32'sd32767)  ? 16'h7FFF :
                                                   ((gt_p[gl] >>> `FRAC_BITS) < -32'sd32768) ? 16'h8000 :
                                                   gt_p[gl][`FRAC_BITS +: `DATA_W];
        end
    endgenerate

    // Datapath registers without reset (written before they are read)
    integer fi;
    always @(posedge clk) begin
        mac_bsum_r <= sat_add_vec(comb_out_r, const_rd_data);
        for (fi = 0; fi < `NUM_LANES; fi = fi + 1)
            comb_out_r[fi*`DATA_W +: `DATA_W] <= sat_shr40($signed(cl_acc_raw_vec[fi*`ACC_W +: `ACC_W]) +
                (!ax_on_r ? 40'sd0 : (fi < 8) ? ax_acc_a[fi % 8] : ax_acc_b[fi % 8]), 5'd11);
        if (state == S_CONV && m2_do_r) m2_res <= sat_add_vec(cl_out_vec, const_rd_data);
        gt_dt <= (state == S_SCAN_LD) ? dt_s : dt_s_n;
        if (ax_on_r || gt_go_g || gt_go_x || sq_g || sq_r)
            for (fi = 0; fi < `NUM_LANES; fi = fi + 1)
                case (gt_sel)
                    2'd0: gt_p[fi] <= $signed(h_new_r[fi*`DATA_W +: `DATA_W]) *
                                      $signed(wx_rd_data2_r[fi*`DATA_W +: `DATA_W]);
                    2'd1: gt_p[fi] <= $signed(y_ssm_word[fi*`DATA_W +: `DATA_W]) *
                                      $signed(silu_z_word[fi*`DATA_W +: `DATA_W]);
                    2'd2: gt_p[fi] <= $signed(sq_op_r[fi*`DATA_W +: `DATA_W]) *
                                      $signed(sq_op_r[fi*`DATA_W +: `DATA_W]);
                    default: gt_p[fi] <= $signed(x_next[fi*`DATA_W +: `DATA_W]) * $signed(gt_dt);
                endcase
        if (gt_go_gd) gt_out   <= gt_sat;
        if (gt_go_xd) xdt_next <= gt_sat;
        if (gt_go_gd)     sq_op_r <= gt_sat;
        else if (rn_ld_r) sq_op_r <= m_rd_data;
        rn_pre_addr_r <= PT_INPUT + t_ch_r + {5'b0, CH_OUT, 1'b0} + {6'b0, g_bank_idx};
        ch_b_op_r <= ax_on_r ? wx_rd_data_r : C_w[tk_s[5]];
        for (fi = 0; fi < `NUM_LANES; fi = fi + 1)
            ch_p[fi] <= $signed(h_new_r[fi*`DATA_W +: `DATA_W]) * $signed(ch_b_op_r[fi*`DATA_W +: `DATA_W]);
        dx_p <= tk_sf[6] ? $signed(tk_x[6]) * $signed(tk_D[6]) : 32'sd0;
        for (fi = 0; fi < 8; fi = fi + 1)
            ch_a[fi] <= ch_p[2*fi] + ch_p[2*fi+1];
        dx_a    <= dx_p;
        ch_b[0] <= ch_a[0] + ch_a[1] + ch_a[2] + ch_a[3];
        ch_b[1] <= ch_a[4] + ch_a[5] + ch_a[6] + ch_a[7];
        dx_b    <= dx_a;
        if (tk_v[9])
            y_acc <= (tk_sf[9] ? 48'sd0 : y_acc) + ch_b[0] + ch_b[1] + dx_b;
        for (fi = 0; fi < 8; fi = fi + 1)
            gt_a[fi] <= gt_p[2*fi] + gt_p[2*fi+1];
        gt_b[0] <= gt_a[0] + gt_a[1] + gt_a[2] + gt_a[3];
        gt_b[1] <= gt_a[4] + gt_a[5] + gt_a[6] + gt_a[7];
        if (state == S_SCAN_LD) begin
            gn_acc <= CLUSTER_ID[0] ? 45'sd0 : $signed({29'd0, gn_eps});   // eps added once, on cluster 0
            rn_acc <= 45'sd0;
        end
        if (sq_g_d[3]) gn_acc <= gn_acc + gt_b[0] + gt_b[1];
        if (sq_r_d[3]) rn_acc <= rn_acc + gt_b[0] + gt_b[1];
        if (ax_t3) begin
            ax_acc_a[7] <= (ax_f3 ? 40'sd0 : ax_acc_a[0]) + ch_b[0] + ch_b[1];
            ax_acc_b[7] <= (ax_f3 ? 40'sd0 : ax_acc_b[0]) + gt_b[0] + gt_b[1];
            for (fi = 0; fi < 7; fi = fi + 1) begin
                ax_acc_a[fi] <= ax_acc_a[fi+1];
                ax_acc_b[fi] <= ax_acc_b[fi+1];
            end
        end

        tk_sf[0] <= (sl_s == 3'd0);
        tk_sl[0] <= (sl_s_w[3:0] == S_CT - 4'd1);
        tk_lg[0] <= (sl_i == NS[7:0] - 8'd1);
        tk_s[0]  <= sl_s;
        tk_c[0]  <= sl_c;
        tk_wa[0] <= h_base[H_ADDR_W-1:0] | {{(H_ADDR_W-8){1'b0}}, sl_i};
        tk_x[0]  <= x_c;
        tk_D[0]  <= D_s;
        tk_g[0]  <= ctr_g;
        for (fi = 1; fi < TK; fi = fi + 1) begin
            tk_sf[fi] <= tk_sf[fi-1];
            tk_sl[fi] <= tk_sl[fi-1];
            tk_lg[fi] <= tk_lg[fi-1];
            tk_s[fi]  <= tk_s[fi-1];
            tk_c[fi]  <= tk_c[fi-1];
            tk_wa[fi] <= tk_wa[fi-1];
            tk_x[fi]  <= tk_x[fi-1];
            tk_D[fi]  <= tk_D[fi-1];
            tk_g[fi]  <= tk_g[fi-1];
        end
    end

    wire gt_go_g_n = (state == S_SCAN) && tk_v[10] && tk_lg[10];
    integer ti;
    always @(posedge clk or posedge rst)
        if (rst) begin
            for (ti = 0; ti < TK; ti = ti + 1) tk_v[ti] <= 1'b0;
            gt_go_g  <= 1'b0;
            gt_go_x  <= 1'b0;
            gt_go_gd <= 1'b0;
            gt_go_xd <= 1'b0;
            gt_sel   <= 2'd3;
            sq_g     <= 1'b0;
            sq_r     <= 1'b0;
            sq_g_d   <= 3'd0;
            sq_r_d   <= 5'd0;
            rn_ld_r     <= 1'b0;
            rn_pre_on_r <= 1'b0;
        end else begin
            gt_go_g  <= gt_go_g_n;
            gt_go_x  <= (state == S_SCAN && ps == 9'd3) || (state == S_SCAN_LD && ld == 5'd5);
            gt_go_gd <= gt_go_g;
            gt_go_xd <= gt_go_x;
            gt_sel   <= ax_on_n ? 2'd0 : gt_go_g_n ? 2'd1 : (gt_go_gd || rn_ld_r) ? 2'd2 : 2'd3;
            sq_g     <= gt_go_gd;
            sq_r     <= rn_ld_r;
            sq_g_d   <= {sq_g_d[2:1], sq_g};
            sq_r_d   <= {sq_r_d[4:1], sq_r};
            rn_pre_on_r <= CLUSTER_ID[0] && ({1'b0, t_cnt} + 11'd2 <= {1'b0, t_last});
            rn_ld_r     <= rn_pre_on_r && (state == S_SCAN) && (ps == 9'd6);
            tk_v[0]  <= (state == S_SCAN) && sl_issue;
            for (ti = 1; ti < TK; ti = ti + 1) tk_v[ti] <= tk_v[ti-1];
        end

    // ------------------------------------------------------------------------
    // GN: s = sum(yg^2) + eps, e = msb(s), k = 2*floor((e-13)/2), r = rom[((s >> k) >> 3) - 1024],
    // out = sat((sat((yg*gamma) >> 11) * r) >> (10 + k/2 - floor(log2(d_inner)/2)))
    // ------------------------------------------------------------------------
    reg  [44:0] gn_s;
    reg  [5:0]  gn_e;
    reg  signed [6:0] gn_half;        // k / 2
    reg  [11:0] gn_idx_r;
    reg  [15:0] gn_r;
    reg  [`NUM_LANES*`DATA_W-1:0] gn_out_reg;
    wire [15:0] gn_rom_data;
    RSqrt_FP_ROM u_gn_rom (.idx(gn_idx_r), .data(gn_rom_data));
    wire [3:0]  gn_fl  = (ch_m_act == 10'd16) ? 4'd4 : 4'd3;            // floor(log2(d_inner) / 2)
    wire signed [7:0] gn_sh_s = 8'sd10 + {gn_half[6], gn_half} - {4'b0, gn_fl};
    reg  [4:0]  gn_sh;                // gn_sh_s, registered in S_GN4
    wire [`NUM_LANES*`DATA_W-1:0] gn_shifted;
    genvar gn_i;
    generate
        for (gn_i = 0; gn_i < `NUM_LANES; gn_i = gn_i + 1) begin : GN_SHIFT
            assign gn_shifted[gn_i*`DATA_W +: `DATA_W] = sat_shr40(cl_acc_raw_vec[gn_i*`ACC_W +: `ACC_W], gn_sh);
        end
    endgenerate
    // Apply slots for group j: 2j read yg and gamma, 2j+3 MUL, 2j+9 latch, 2j+10 MUL by r (raw product in acc),
    // 2j+15 shift + saturate acc_raw, 2j+16 write
    wire [8:0]  gn_n      = {1'b0, ch_m_act[8:1]};
    wire        gn_do_rd  = !ps[0]                && ({1'b0, ps[8:1]}      < gn_n);
    wire [8:0]  gn_grp_rd = {ps[8:1],      CLUSTER_ID[0]};
    reg  [16:1] gn_sr;                // gn_sr[k]: slot 2j+k of some group j is due
    reg  [7:0]  gn_wr_j;
    wire        gn_do_m1  = gn_sr[3];
    wire        gn_do_p1  = gn_sr[9];
    wire        gn_do_m2  = gn_sr[10];
    wire        gn_do_cap = gn_sr[15];
    wire        gn_do_wr  = gn_sr[16];
    always @(posedge clk or posedge rst)
        if (rst) gn_sr <= 16'd0;
        else     gn_sr <= {gn_sr[15:1], (state == S_GN6) && gn_do_rd};
    assign gn_part = gn_acc;
    wire [8:0]  gn_ps_end = {gn_n[7:0], 1'b0} + 9'd14;

    // ------------------------------------------------------------------------
    // FIN, one pass after the t-loop: final = sat(relu(sat(sat(inc*scale >> 11) + shift)) + relu(m8)), written
    // over inc in PT_INC_OUT. Item i = t*n + j covers this cluster's group 2j + CLUSTER_ID (n = CH_OUT/2), so both
    // inputs sit at base + 2i + CLUSTER_ID. MaxPool2: even t keeps final in fin_prev_r[j], odd t writes
    // max(final[t-1], final[t]) to slot t>>1, which is always already consumed.
    // Slots of item i, q = 2i (counter ctr_k):
    //   q    read inc          q+1  read scale         q+3  PE MUL inc*scale
    //   q+8  read shift        q+9  read m8, latch the product
    //   q+10 fin_a             q+12 final (m8 on the bus)      q+13 pool, GAP accumulate       q+14 write
    // ------------------------------------------------------------------------
    wire        fin_n4    = (CH_OUT == 9'd8);                            // n = 4, else 2
    wire [13:0] fin_items = fin_n4 ? {2'b0, T_MAX, 2'b0} : {3'b0, T_MAX, 1'b0};
    function automatic fin_ev;                                           // slot k of some item is due at q
        input [12:0] q;
        input [12:0] k;
        input [13:0] items;
        reg   [12:0] e;
        begin
            e = q - k;
            fin_ev = (q >= k) && !e[0] && ({2'b0, e[12:1]} < items);
        end
    endfunction
    wire [12:0] fin_e1  = ctr_k - 13'd1;
    wire [12:0] fin_e8  = ctr_k - 13'd8;
    wire [12:0] fin_e9  = ctr_k - 13'd9;
    wire [12:0] fin_e12 = ctr_k - 13'd12;
    wire [1:0]  fin_j1    = fin_n4 ? fin_e1[2:1]  : {1'b0, fin_e1[1]};
    wire [1:0]  fin_j8    = fin_n4 ? fin_e8[2:1]  : {1'b0, fin_e8[1]};
    wire [1:0]  fin_j12   = fin_n4 ? fin_e12[2:1] : {1'b0, fin_e12[1]};
    wire        fin_odd12 = fin_n4 ? fin_e12[3] : fin_e12[2];            // t[0]
    wire [9:0]  fin_t12   = fin_n4 ? fin_e12[12:3] : fin_e12[11:2];
    wire [8:0]  fin_g1    = {6'b0, fin_j1,  CLUSTER_ID[0]};
    wire [8:0]  fin_g8    = {6'b0, fin_j8,  CLUSTER_ID[0]};
    wire [8:0]  fin_g12   = {6'b0, fin_j12, CLUSTER_ID[0]};
    wire [14:0] fin_w0    = {2'b0, ctr_k[12:1], 1'b0}   + {14'b0, CLUSTER_ID[0]};
    wire [14:0] fin_w9    = {2'b0, fin_e9[12:1], 1'b0}  + {14'b0, CLUSTER_ID[0]};
    wire [14:0] fin_w12   = {2'b0, fin_e12[12:1], 1'b0} + {14'b0, CLUSTER_ID[0]};
    wire [14:0] fin_pool_w = (fin_n4 ? {3'b0, fin_t12[9:1], 3'b0} : {4'b0, fin_t12[9:1], 2'b0}) + {6'b0, fin_g12};
    wire [12:0] fin_end   = {fin_items[11:0], 1'b0} + 13'd12;
    // Slots 13 and 14 of an item are its slot 12 one and two cycles later (ctr_k steps by 1 in S_FIN and is 0
    // before it), so their events, the item's group and t parity, the address and final are registered from slot 12.
    reg         fin_ev13, fin_odd13, fin_wr_ev13, fin_wr_ev, fin_gap_ev;
    reg  [14:0] fin_wr_addr13, fin_wr_addr;
    reg  [1:0]  fin_j13;
    reg  [`NUM_LANES*`DATA_W-1:0] fin_p, fin_a, fin_res, fin_final_r;
    reg  [`NUM_LANES*`DATA_W-1:0] fin_prev_r [0:3];
    wire [`NUM_LANES*`DATA_W-1:0] fin_final = sat_add_vec(fin_a, relu_vec(m_rd_data));
    reg  [12:1] fin_sr;                                                  // fin_sr[k]: slot k of some item is due
    wire        fin_ev0 = fin_ev(ctr_k, 13'd0, fin_items);
    always @(posedge clk or posedge rst)
        if (rst) begin
            fin_ev13      <= 1'b0;
            fin_odd13     <= 1'b0;
            fin_wr_ev13   <= 1'b0;
            fin_wr_ev     <= 1'b0;
            fin_gap_ev    <= 1'b0;
            fin_wr_addr13 <= 15'd0;
            fin_wr_addr   <= 15'd0;
            fin_j13       <= 2'd0;
            fin_final_r   <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_sr        <= 12'd0;
        end else begin
            fin_sr        <= {fin_sr[11:1], (state == S_FIN) && fin_ev0};
            fin_ev13      <= fin_sr[12];
            fin_odd13     <= fin_odd12;
            fin_wr_ev13   <= fin_sr[12] && (!POOL_OUT || fin_odd12);
            fin_wr_ev     <= fin_wr_ev13;
            fin_gap_ev    <= fin_sr[12] && GAP_EN;
            fin_wr_addr13 <= PT_INC_OUT + (POOL_OUT ? fin_pool_w : fin_w12);
            fin_wr_addr   <= fin_wr_addr13;
            fin_j13       <= fin_j12;
            fin_final_r   <= fin_final;
        end

    // ------------------------------------------------------------------------
    // GAP: each cluster sums its own groups' FIN outputs over T in 25-bit lanes (|sum| <= T*32767 < 2^24 for
    // T <= 512); S_GAP then emits sat16((sum * GAP_INV) >>> 15) one lane per cycle (lane select, product, saturate).
    // ------------------------------------------------------------------------
    localparam GAP_W = 25;
    reg  [`NUM_LANES*GAP_W-1:0]  gap_acc [0:3];
    reg  [4:0]                   gap_lane;
    reg  [GAP_W-1:0]             gap_sum_r;
    reg  signed [GAP_W+17-1:0]   gap_prod;
    reg  [`NUM_LANES*`DATA_W-1:0] gap_word;
    wire [GAP_W-1:0] gap_lane_sum = gap_acc[ctr_g[2:1]][gap_lane[3:0]*GAP_W +: GAP_W];
    wire signed [GAP_W+17-1:0] gap_shifted = gap_prod >>> 15;
    wire [`DATA_W-1:0] gap_sat16 = (gap_shifted > 32767)  ? 16'h7FFF :
                                   (gap_shifted < -32768) ? 16'h8000 : gap_shifted[`DATA_W-1:0];
    function [`NUM_LANES*GAP_W-1:0] gap_add;
        input [`NUM_LANES*GAP_W-1:0] acc;
        input [`NUM_LANES*`DATA_W-1:0] x;
        integer i;
        for (i = 0; i < `NUM_LANES; i = i + 1)
            gap_add[i*GAP_W +: GAP_W] = acc[i*GAP_W +: GAP_W] +
                {{(GAP_W-`DATA_W){x[(i+1)*`DATA_W-1]}}, x[i*`DATA_W +: `DATA_W]};
    endfunction

    assign done_stage = (state == S_BARRIER) && !mp_wr;      // the MP3 write leaves one cycle after the state
    assign done_all   = (state == S_DONE);

    // ------------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state          <= S_IDLE;
            cur_stage      <= 5'd0;
            t_cnt          <= 10'd0;
            wait_cnt       <= 3'd0;
            ctr_g          <= 9'd0;
            ctr_k          <= 13'd0;
            h_init_cnt     <= 16'd0;
            next_stage_q   <= 5'd0;
            next_state_q   <= S_IDLE;
            ps             <= 9'd0;
            m_rd_addr      <= 15'd0;
            m_we           <= 1'b0;
            m_wr_addr      <= 15'd0;
            m_wr_data      <= {`NUM_LANES*`DATA_W{1'b0}};
            w_rd_addr      <= 15'd0;
            w_rd_addr2     <= 15'd0;
            const_rd_addr  <= 15'd0;
            cl_op_mode     <= `MAMBA_PE_IDLE;
            cl_clear_acc   <= 1'b0;
            cl_in_W1_vec   <= {`NUM_LANES*`DATA_W{1'b0}};
            cl_in_H_ext    <= {`NUM_LANES*`DATA_W{1'b0}};
            cl_in_W2_vec   <= {`NUM_LANES*`DATA_W{1'b0}};
            cl_in_X_vec    <= {`NUM_LANES*`DATA_W{1'b0}};
            h_from_rf_r    <= 1'b0;
            h_rd_addr_r    <= {H_ADDR_W{1'b0}};
            h_wr_en_r      <= 1'b0;
            h_wr_addr_r    <= {H_ADDR_W{1'b0}};
            h_wr_data_r    <= {`NUM_LANES*`DATA_W{1'b0}};
            rsqrt_idx_r    <= 13'd0;
            rn_scale       <= 16'd0;
            rn_hoist       <= 1'b0;
            tin_r          <= 15'd0;
            tout_r         <= 15'd0;
            mp_ph          <= 2'd0;
            mp_fv          <= 1'b0;
            mp_drain       <= 1'b0;
            mp_acc         <= {`NUM_LANES*`DATA_W{1'b0}};
            mp_wr          <= 1'b0;
            mp_wr_addr     <= 15'd0;
            mp_wa_pend     <= 15'd0;
            ovl_pend       <= 1'b0;
            ovl_cnt        <= 3'd0;
            ovl_addr       <= 15'd0;
            ld             <= 5'd0;
            dt_word        <= {`NUM_LANES*`DATA_W{1'b0}};
            dA_word        <= {`NUM_LANES*`DATA_W{1'b0}};
            D_word         <= {`NUM_LANES*`DATA_W{1'b0}};
            x_word         <= {`NUM_LANES*`DATA_W{1'b0}};
            xdt_word       <= {`NUM_LANES*`DATA_W{1'b0}};
            x_next         <= {`NUM_LANES*`DATA_W{1'b0}};
            silu_z_word    <= {`NUM_LANES*`DATA_W{1'b0}};
            y_ssm_word     <= {`NUM_LANES*`DATA_W{1'b0}};
            h_new_r        <= {`NUM_LANES*`DATA_W{1'b0}};
            gn_s           <= 45'd0;
            gn_e           <= 6'd0;
            gn_half        <= 7'sd0;
            gn_sh          <= 5'd0;
            gn_wr_j        <= 8'd0;
            gn_idx_r       <= 12'd0;
            gn_r           <= 16'd0;
            gn_out_reg     <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_p          <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_a          <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_res        <= {`NUM_LANES*`DATA_W{1'b0}};
            gap_acc[0]     <= {`NUM_LANES*GAP_W{1'b0}};
            gap_acc[1]     <= {`NUM_LANES*GAP_W{1'b0}};
            gap_acc[2]     <= {`NUM_LANES*GAP_W{1'b0}};
            gap_acc[3]     <= {`NUM_LANES*GAP_W{1'b0}};
            gap_lane       <= 5'd0;
            gap_sum_r      <= {GAP_W{1'b0}};
            gap_prod       <= {(GAP_W+17){1'b0}};
            gap_word       <= {`NUM_LANES*`DATA_W{1'b0}};
        end else begin
            cl_op_mode   <= `MAMBA_PE_IDLE;
            cl_clear_acc <= 1'b0;
            m_we         <= 1'b0;
            h_wr_en_r    <= 1'b0;
            mp_wr        <= 1'b0;

            // Scale of x[t+2] from rn_acc, one and two cycles after each of its accumulates (the last one wins)
            if (sq_r_d[4]) rsqrt_idx_r <= rn_clip(rn_acc >>> rn_shift);
            if (sq_r_d[5]) rn_scale    <= rsqrt_data;

            case (state)
                S_IDLE, S_DONE: begin
                    if (start) begin
                        t_cnt      <= 10'd0;
                        ctr_g      <= {8'd0, CLUSTER_ID[0]};
                        ctr_k      <= 13'd0;
                        h_init_cnt <= 16'd0;
                        gap_acc[0] <= {`NUM_LANES*GAP_W{1'b0}};
                        gap_acc[1] <= {`NUM_LANES*GAP_W{1'b0}};
                        gap_acc[2] <= {`NUM_LANES*GAP_W{1'b0}};
                        gap_acc[3] <= {`NUM_LANES*GAP_W{1'b0}};
                        if (PHASE_INC) begin
                            // through the barrier, whose release sets up the time-split bases
                            next_stage_q <= ENC_EN ? STG_ENC : STG_P1;
                            next_state_q <= S_MAC1;
                            state        <= S_BARRIER;
                        end else begin
                            cur_stage <= STG_M6;     // not M8, so the barrier after H_INIT does not advance t
                            state     <= S_H_INIT;
                        end
                    end
                end

                // Zero the scan state (URAM, no initial values)
                S_H_INIT: begin
                    if ({2'b0, h_init_cnt} < h_init_limit) begin
                        h_wr_en_r   <= 1'b1;
                        h_wr_addr_r <= h_init_cnt[H_ADDR_W-1:0];
                        h_wr_data_r <= {`NUM_LANES*`DATA_W{1'b0}};
                        h_init_cnt  <= h_init_cnt + 16'd1;
                    end else begin
                        h_init_cnt   <= 16'd0;
                        next_stage_q <= STG_RN;
                        next_state_q <= S_RN1;
                        state        <= S_BARRIER;
                    end
                end

                S_BARRIER: begin
                    if (bar_go) begin
                        cur_stage <= next_stage_q;
                        state     <= next_state_q;
                        if (cur_stage == STG_M8 && next_stage_q == STG_RN)
                            t_cnt <= t_cnt + 10'd1;
                        ctr_g    <= {8'd0, CLUSTER_ID[0]};
                        ctr_k    <= 13'd0;
                        wait_cnt <= 3'd0;
                        if (mac_tsplit_of(next_stage_q)) begin
                            ctr_g  <= 9'd0;
                            t_cnt  <= {9'd0, CLUSTER_ID[0]};
                            tin_r  <= tsplit_in_base_of(next_stage_q) +
                                      (CLUSTER_ID[0] ? {6'b0, tsplit_in_wpt_of(next_stage_q, CH_IN, CH_OUT)} : 15'd0);
                            tout_r <= tsplit_out_base_of(next_stage_q, CH_OUT) +
                                      (CLUSTER_ID[0] ? {6'b0, tsplit_out_wpt_of(next_stage_q, CH_IN, CH_OUT)} : 15'd0);
                        end
                    end
                end

                // ------------------------------------------------------------
                // MAC chain
                // ------------------------------------------------------------
                S_MAC1: begin
                    m_rd_addr  <= mac_rd_addr_k0;
                    w_rd_addr  <= mac_w_addr_k0;
                    w_rd_addr2 <= mac_w_addr_k0 + 15'd1;
                    if (mac_bias_en)
                        const_rd_addr <= mac_bias_base + {6'b0, ctr_g};
                    state      <= S_MAC2;
                end

                S_MAC2: begin
                    if (wait_cnt == 3'd0) begin
                        wait_cnt   <= 3'd1;
                        w_rd_addr  <= mac_w_addr_k2;
                        w_rd_addr2 <= mac_w_addr_k2 + 15'd1;
                        m_rd_addr  <= ax_slot_r ? ax_rd_addr_r : mac_rd_addr_k2;
                        ctr_k      <= 13'd0;
                    end else begin
                        wait_cnt   <= 3'd0;
                        m_rd_addr  <= mac_rd_addr_k2;
                        w_rd_addr  <= mac_w_addr_k4;
                        w_rd_addr2 <= mac_w_addr_k4 + 15'd1;
                        state      <= S_MAC3;
                    end
                end

                S_MAC3: begin
                    cl_in_W1_vec <= w_rd_data_r;
                    cl_in_W2_vec <= w_rd_data2_r;
                    cl_in_H_ext  <= mac_y1;
                    cl_in_X_vec  <= mac_y2;
                    cl_op_mode   <= `MAMBA_PE_MAC2;
                    cl_clear_acc <= 1'b1;
                    m_rd_addr    <= mac_rd_addr_k6;
                    w_rd_addr    <= mac_w_addr_k6;
                    w_rd_addr2   <= mac_w_addr_k6 + 15'd1;
                    ctr_k        <= ctr_k_p2;
                    state        <= S_MAC4;
                end

                S_MAC4: begin
                    // No MAC on the exit cycle: the look-ahead has already fetched the next group's weights.
                    if (!mac_k_end) begin
                        cl_op_mode   <= `MAMBA_PE_MAC2;
                        cl_in_W1_vec <= w_rd_data_r;
                        cl_in_W2_vec <= w_rd_data2_r;
                        cl_in_H_ext  <= mac_y1;
                        cl_in_X_vec  <= mac_y2;
                    end
                    m_rd_addr  <= ax_slot_r ? ax_rd_addr_r : mac_rd_addr_k6;
                    w_rd_addr  <= mac_w_addr_k6;
                    w_rd_addr2 <= mac_w_addr_k6 + 15'd1;
                    ctr_k      <= ctr_k_p2;
                    // Deferred write of the previous group. The next group stays in S_MAC4 until at least
                    // cycle 4 + (len-2)/2 >= 11 after the exit (len >= 16), so this slot always comes.
                    if (ovl_pend && ovl_cnt == 3'd6) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= ovl_addr;
                        m_wr_data <= ovl_data;
                    end
                    if (mac_ovl_exit) begin
                        ovl_pend <= 1'b1;
                        ovl_cnt  <= 3'd0;
                        ovl_addr <= mac_wr_addr;
                        if (mac_tsplit) begin
                            ctr_g <= mac_g_last_in_t ? 9'd0 : ctr_g + 9'd1;
                            if (mac_g_last_in_t) begin
                                t_cnt  <= t_cnt + 10'd2;
                                tin_r  <= tin_next;
                                tout_r <= tout_next;
                            end
                        end else begin
                            ctr_g <= ctr_g_p2;
                        end
                        ctr_k      <= 13'd0;
                        m_rd_addr  <= mac_rd_addr_g0;
                        w_rd_addr  <= w_grp_base_next_r;
                        w_rd_addr2 <= w_grp_base_next_r + 15'd1;
                        state      <= S_MAC2;
                    end else if (mac_k_end) begin
                        state <= S_MAC5;
                    end
                end

                // Last group: drain, write (bias jobs one cycle later, from mac_bsum_r), then the barrier
                S_MAC5: begin
                    if (wait_cnt != (mac_bias_en ? 3'd5 : 3'd4)) begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end else begin
                        wait_cnt  <= 3'd0;
                        m_we      <= 1'b1;
                        m_wr_addr <= mac_wr_addr;
                        m_wr_data <= mac_bias_en ? mac_bsum_r : comb_out_r;
                        if (mac_last_iter) begin
                            ctr_g        <= {8'd0, CLUSTER_ID[0]};
                            ctr_k        <= 13'd0;
                            next_stage_q <= mac_next_stage;
                            next_state_q <= mac_next_state;
                            // Cluster 1 has one in_proj group less (odd group count): it computes RN of t+1 now.
                            // For t >= 1 its scale came from the scan of t-1, so only the apply is left.
                            if (CLUSTER_ID[0] && cur_stage == STG_M1A && t_cnt != t_last) begin
                                rn_hoist <= 1'b1;
                                ps       <= 9'd0;
                                state    <= (t_cnt != 10'd0) ? S_RN7 : S_RN1;
                            end else
                                state    <= S_BARRIER;
                        end else begin
                            if (mac_tsplit) begin
                                ctr_g <= mac_g_last_in_t ? 9'd0 : ctr_g + 9'd1;
                                if (mac_g_last_in_t) begin
                                    t_cnt  <= t_cnt + 10'd2;
                                    tin_r  <= tin_next;
                                    tout_r <= tout_next;
                                end
                                if (mac_bias_en)
                                    const_rd_addr <= mac_bias_base + (mac_g_last_in_t ? 15'd0 : {6'b0, ctr_g} + 15'd1);
                            end else begin
                                ctr_g <= ctr_g_p2;
                            end
                            ctr_k      <= 13'd0;
                            m_rd_addr  <= mac_rd_addr_g0;
                            w_rd_addr  <= w_grp_base_next_r;
                            w_rd_addr2 <= w_grp_base_next_r + 15'd1;
                            state      <= S_MAC2;
                        end
                    end
                end

                // ------------------------------------------------------------
                // MP3, one read per cycle: item k reads t, t-1, t+1 at phases 0-2 of cycles 3k..3k+2 and steps
                // (g, t) at phase 2; its words arrive at 3k+3..3k+5 and are folded under item k+1's reads.
                // ------------------------------------------------------------
                S_MP3: begin
                    mp_ph <= (mp_ph == 2'd2) ? 2'd0 : mp_ph + 2'd1;
                    if (!mp_drain)
                        case (mp_ph)
                            2'd0: m_rd_addr <= mp_cur_addr;
                            2'd1: m_rd_addr <= mp_prev_addr;
                            default: begin
                                m_rd_addr  <= mp_next_addr;
                                mp_wa_pend <= tout_r + {6'b0, ctr_g};
                                mp_fv      <= 1'b1;
                                if (mac_g_last_in_t) begin
                                    ctr_g <= 9'd0;
                                    if (mac_t_last)
                                        mp_drain <= 1'b1;
                                    else begin
                                        t_cnt  <= t_cnt + 10'd2;
                                        tin_r  <= tin_next;
                                        tout_r <= tout_next;
                                    end
                                end else begin
                                    ctr_g <= ctr_g + 9'd1;
                                end
                            end
                        endcase
                    if (mp_fv)
                        case (mp_ph)
                            2'd0: mp_acc <= m_rd_data;
                            2'd1: mp_acc <= max_vec(mp_acc, m_rd_data);
                            default: begin
                                mp_acc     <= max_vec(mp_acc, m_rd_data);
                                mp_wr      <= 1'b1;
                                mp_wr_addr <= mp_wa_pend;
                                if (mp_drain) begin
                                    mp_fv        <= 1'b0;
                                    mp_drain     <= 1'b0;
                                    next_stage_q <= mac_next_stage;
                                    next_state_q <= mac_next_state;
                                    state        <= S_BARRIER;
                                end
                            end
                        endcase
                end

                // ------------------------------------------------------------
                // RN
                // ------------------------------------------------------------
                S_RN1: begin
                    ctr_k     <= 13'd0;
                    m_rd_addr <= rn_sq_addr_k0;
                    if (!rn_hoist && t_cnt != 10'd0) begin      // already computed by cluster 1
                        next_stage_q <= STG_M1A;
                        next_state_q <= S_MAC1;
                        state        <= S_BARRIER;
                    end else
                        state <= S_RN2;
                end

                // Words 0, 1, 2 are issued in RN1, RN2, RN2; RN3 then stays 3 words ahead of the word it consumes.
                S_RN2: begin
                    if (wait_cnt != 3'd1) begin
                        wait_cnt  <= wait_cnt + 3'd1;
                        m_rd_addr <= rn_sq_addr_k1;
                    end else begin
                        wait_cnt  <= 3'd0;
                        m_rd_addr <= rn_sq_addr_k2;
                        state     <= S_RN3;
                    end
                end

                S_RN3: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (ctr_k == 13'd0);
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= m_rd_data;
                    m_rd_addr    <= rn_sq_addr_k3;
                    ctr_k        <= ctr_k_p1;
                    if (ctr_k == {4'b0, CH_OUT} - 13'd1) state <= S_RN4;
                end

                S_RN4: begin                                    // PE drain + reduce tree
                    if (wait_cnt == 3'd5) begin
                        wait_cnt <= 3'd0;
                        state    <= S_RN5;
                    end else
                        wait_cnt <= wait_cnt + 3'd1;
                end

                S_RN5: begin
                    rsqrt_idx_r <= rn_clip(sum_d_wide >>> rn_shift);
                    state       <= S_RN6;
                end

                S_RN6: begin
                    rn_scale <= rsqrt_data;
                    ps       <= 9'd0;
                    state    <= S_RN7;
                end

                S_RN7: begin
                    ps <= ps + 9'd1;
                    if (rn_do_rd) begin
                        m_rd_addr     <= PT_INPUT + rn_t_off + {6'b0, rn_grp_rd};
                        const_rd_addr <= C_W_NORM_BASE + {6'b0, rn_grp_rd};
                    end
                    if (rn_do_m1) begin
                        cl_op_mode   <= `MAMBA_PE_MUL;
                        cl_in_W1_vec <= m_rd_data;
                        cl_in_H_ext  <= const_rd_data;
                    end
                    if (rn_do_l) h_new_r <= cl_out_vec;
                    if (rn_do_m2) begin
                        cl_op_mode   <= `MAMBA_PE_MUL;
                        cl_in_W1_vec <= h_new_r;
                        cl_in_H_ext  <= {`NUM_LANES{rn_scale}};
                    end
                    if (rn_do_wr) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= rn_xn_base + {6'b0, rn_grp_wr};
                        m_wr_data <= cl_out_vec;
                    end
                    if (ps == rn_ps_end) begin
                        ctr_g    <= {8'd0, CLUSTER_ID[0]};
                        ctr_k    <= 13'd0;
                        rn_hoist <= 1'b0;
                        if (!rn_hoist) begin                    // hoisted: in_proj already set the next stage
                            next_stage_q <= STG_M1A;
                            next_state_q <= S_MAC1;
                        end
                        state <= S_BARRIER;
                    end
                end

                // ------------------------------------------------------------
                // M2 conv
                // ------------------------------------------------------------
                S_CONV0: begin
                    ps    <= 9'd0;
                    state <= S_CONV;
                end

                S_CONV: begin
                    ps <= ps + 9'd1;
                    if (m2_do_f) begin
                        m_rd_addr <= fbx_r[m2_slotf] + {6'b0, m2_grp_f};
                        w_rd_addr <= W_CONV_BASE + {4'b0, m2_grp_f, m2_tapf};
                    end
                    if (m2_do_mac) begin
                        cl_op_mode   <= `MAMBA_PE_MAC;
                        cl_clear_acc <= (m2_tapm == 2'd0);
                        cl_in_W1_vec <= w_rd_data_r;
                        cl_in_H_ext  <= m2_pad ? {`NUM_LANES*`DATA_W{1'b0}} : m_rd_data;
                    end
                    if (m2_do_b) const_rd_addr <= C_B_CONV_BASE + {6'b0, m2_grp_b};
                    if (m2_do_w) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_XCONV + {6'b0, m2_grp_w};
                        m_wr_data <= m2_res;
                    end
                    if (m2_do_s) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_XBC + {6'b0, m2_grp_s};
                        m_wr_data <= silu_out_w;
                    end
                    if (ps == m2_ps_end) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_k        <= 13'd0;
                        wait_cnt     <= 3'd0;
                        next_stage_q <= STG_M6;
                        next_state_q <= S_SCAN_LD;
                        state        <= S_BARRIER;
                    end
                end

                // ------------------------------------------------------------
                // M6 load phase: word i read at ld = i, on the bus at i + 3. Both clusters compute dt and dA.
                //   ld 0  dt_raw + const dt_bias      ld 1  x          ld 2 .. S+1  B[s]     ld S+2 .. 2S+1  C[s]
                //   ld 3  dt_sum, const A             ld 4  dt = softplus(dt_sum)
                //   ld 5  dt written to PT_DT; PE MUL dt*A, const D; first group's x*dt on the gate row (5 .. 7)
                //   ld 7  D                           ld 11 dA = exp(dt*A), written to PT_DA
                // ------------------------------------------------------------
                S_SCAN_LD: begin
                    ld <= ld + 5'd1;
                    if (ld == 5'd0) begin
                        m_rd_addr     <= dt_raw_addr;
                        const_rd_addr <= C_DT_BIAS_BASE;
                    end else if (ld == 5'd1)
                        m_rd_addr <= PT_XBC + {6'b0, ctr_g};
                    else if (ld <= {S_CT, 1'b0} + 5'd1)
                        m_rd_addr <= PT_XBC + {5'b0, ch_m_act} + {10'b0, ld} - 15'd2;
                    if (ld == 5'd3) begin
                        dt_sum        <= sat_add_vec(m_rd_data, const_rd_data);
                        const_rd_addr <= C_A_BASE;
                    end
                    if (ld == 5'd4) begin
                        x_word    <= m_rd_data;
                        x_next    <= m_rd_data;
                        dt_word   <= sp_out_w;
                    end
                    if (ld == 5'd5) begin
                        m_we          <= 1'b1;
                        m_wr_addr     <= PT_DT;
                        m_wr_data     <= dt_word;
                        cl_op_mode    <= `MAMBA_PE_MUL;
                        cl_in_W1_vec  <= dt_word;
                        cl_in_H_ext   <= const_rd_data;
                        const_rd_addr <= C_D_BASE;
                    end
                    if (ld == 5'd7) D_word <= const_rd_data;
                    if (ld >= 5'd5 && ld <= {1'b0, S_CT} + 5'd4)
                        B_w[ld - 5'd5] <= m_rd_data;
                    else if (ld >= {1'b0, S_CT} + 5'd5 && ld <= {S_CT, 1'b0} + 5'd4)
                        C_w[ld - {1'b0, S_CT} - 5'd5] <= m_rd_data;
                    if (ld == 5'd11) begin
                        dA_word   <= exp_out_w;
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_DA;
                        m_wr_data <= exp_out_w;
                    end
                    if (ld == LD_END) begin
                        xdt_word <= xdt_next;
                        ld       <= 5'd0;
                        ps       <= 9'd0;
                        state    <= S_SCAN;
                    end
                end

                // ------------------------------------------------------------
                // M6 slot loop, one slot per cycle (ps = cycle within the group):
                //   ps = i      read h[c][s]                        (on the PE input 3 cycles later)
                //   ps = i+2    MUL2 dA*h, xdt[c]*B[s]; token pushed
                //   ps = i+8    h_new = sat(dA*h + xdt*B) latched
                //   ps = i+9    token stage 6: h_new written back; C[s]*h_new (+ D*x[c] if s = 0) on the fabric row
                //   ps = i+13   stage 10: y accumulated; at the last s, y[c] = sat(sum >> 11)
                //   last slot:  stage 11 y_ssm out + gate product, stage 12 gate saturate, stage 13 y_gated out
                // Under each group: next group's x read at ps 0 (latched 3), its x*dt on the gate row at 4/5;
                // this group's z read at ps 12, SiLU latched at 15.
                // ------------------------------------------------------------
                S_SCAN: begin
                    ps          <= ps + 9'd1;
                    h_new_r     <= h_new_w;
                    h_from_rf_r <= sl_issue;
                    if (ps == 9'd0)  m_rd_addr   <= PT_XBC + {6'b0, ctr_g_p2};
                    if (ps == 9'd3)  x_next      <= m_rd_data;
                    if (ps == 9'd12) m_rd_addr   <= fb_r[t_cnt[1:0]] + {6'b0, ctr_g};
                    if (CLUSTER_ID[0] && ps == 9'd4) m_rd_addr <= rn_pre_addr_r;   // x[t+2] word for rn_acc
                    if (ps == 9'd15) silu_z_word <= silu_out_w;
                    if (ps < NS)
                        h_rd_addr_r <= h_base[H_ADDR_W-1:0] | {{(H_ADDR_W-8){1'b0}}, ps[7:0]};
                    if (sl_issue) begin
                        cl_op_mode   <= `MAMBA_PE_MUL2;
                        cl_in_W1_vec <= {`NUM_LANES{dA_s}};
                        cl_in_W2_vec <= {`NUM_LANES{xdt_c}};
                        cl_in_X_vec  <= B_w[sl_s];
                    end
                    if (tk_v[6]) begin
                        h_wr_en_r   <= 1'b1;
                        h_wr_addr_r <= tk_wa[6];
                        h_wr_data_r <= h_new_r;
                    end
                    if (tk_v[10] && tk_sl[10]) y_ssm_word[{tk_c[10], 4'b0} +: `DATA_W] <= y_sat;
                    if (tk_v[11] && tk_lg[11]) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_Y_SSM + {6'b0, tk_g[11]};
                        m_wr_data <= y_ssm_word;
                    end
                    if (tk_v[13] && tk_lg[13]) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_Y_GATED + {6'b0, tk_g[13]};
                        m_wr_data <= gt_out;
                    end
                    if (ps == GP - 9'd1 && scan_more_r) begin
                        ctr_g    <= ctr_g_p2;               // B, C, dt, dA, D are the same for all groups
                        x_word   <= x_next;
                        xdt_word <= xdt_next;
                        ps       <= 9'd0;
                    end
                    // End when the last y_gated square is in gn_acc (written 3 cycles after y_gated)
                    if (!scan_more_r && ps == NS + 9'd18) begin
                        h_from_rf_r  <= 1'b0;
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_k        <= 13'd0;
                        wait_cnt     <= 3'd0;
                        next_stage_q <= STG_GN;
                        next_state_q <= S_GN1;
                        state        <= S_BARRIER;
                    end
                end

                // ------------------------------------------------------------
                // GN: the two clusters' sums of squares (+ eps, from the scan), rsqrt, then the pipelined apply
                // ------------------------------------------------------------
                S_GN1: begin
                    gn_s  <= gn_acc + peer_gn_part;
                    state <= S_GN2;
                end
                S_GN2: begin
                    gn_e  <= msb45(gn_s);
                    state <= S_GN3;
                end
                S_GN3: begin
                    gn_half <= ($signed({1'b0, gn_e}) - 7'sd13) >>> 1;
                    state   <= S_GN4;
                end
                S_GN4: begin
                    if (gn_half < 0) gn_idx_r <= ({gn_s[14:0], 2'b00} >> 3) - 12'd1024;
                    else             gn_idx_r <= ((gn_s >> {gn_half[4:0], 1'b0}) >> 3) - 12'd1024;
                    gn_sh <= gn_sh_s[4:0];
                    state <= S_GN5;
                end
                S_GN5: begin
                    gn_r    <= gn_rom_data;
                    gn_wr_j <= 8'd0;
                    ps      <= 9'd0;
                    state <= S_GN6;
                end
                S_GN6: begin
                    ps <= ps + 9'd1;
                    if (gn_do_rd) begin
                        m_rd_addr     <= PT_Y_GATED     + {6'b0, gn_grp_rd};
                        const_rd_addr <= C_W_GNORM_BASE + {6'b0, gn_grp_rd};
                    end
                    if (gn_do_m1) begin
                        cl_op_mode   <= `MAMBA_PE_MUL;
                        cl_in_W1_vec <= m_rd_data;
                        cl_in_H_ext  <= const_rd_data;
                    end
                    if (gn_do_p1) h_new_r <= cl_out_vec;
                    if (gn_do_m2) begin
                        cl_op_mode   <= `MAMBA_PE_MUL;
                        cl_in_W1_vec <= h_new_r;
                        cl_in_H_ext  <= {`NUM_LANES{gn_r}};
                    end
                    if (gn_do_cap) gn_out_reg <= gn_shifted;
                    if (gn_do_wr) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_Y_NORM + {6'b0, gn_wr_j, CLUSTER_ID[0]};
                        gn_wr_j   <= gn_wr_j + 8'd1;
                        m_wr_data <= gn_out_reg;
                    end
                    if (ps == gn_ps_end) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_k        <= 13'd0;
                        wait_cnt     <= 3'd0;
                        next_stage_q <= STG_M8;
                        next_state_q <= S_MAC1;
                        state        <= S_BARRIER;
                    end
                end

                // ------------------------------------------------------------
                // FIN pass
                // ------------------------------------------------------------
                S_FIN: begin
                    ctr_k <= ctr_k + 13'd1;
                    if (fin_ev0)
                        m_rd_addr <= PT_INC_OUT + fin_w0;
                    if (fin_sr[1])
                        const_rd_addr <= C_INC_SCALE_BASE + {6'b0, fin_g1};
                    if (fin_sr[3]) begin
                        cl_op_mode   <= `MAMBA_PE_MUL;
                        cl_in_W1_vec <= m_rd_data;
                        cl_in_H_ext  <= const_rd_data;
                    end
                    if (fin_sr[8])
                        const_rd_addr <= C_INC_SHIFT_BASE + {6'b0, fin_g8};
                    if (fin_sr[9]) begin
                        m_rd_addr <= PT_MAMBA_OUT + fin_w9;
                        fin_p     <= cl_out_vec;
                    end
                    if (fin_sr[10])
                        fin_a <= relu_vec(sat_add_vec(fin_p, const_rd_data));
                    if (fin_ev13) begin
                        if (POOL_OUT && !fin_odd13) fin_prev_r[fin_j13] <= fin_final_r;
                        fin_res <= (POOL_OUT && fin_odd13) ? max_vec(fin_prev_r[fin_j13], fin_final_r) : fin_final_r;
                    end
                    if (fin_gap_ev) gap_acc[fin_j13] <= gap_add(gap_acc[fin_j13], fin_final_r);
                    if (fin_wr_ev) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= fin_wr_addr;
                        m_wr_data <= fin_res;
                    end
                    if (ctr_k == fin_end) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_k        <= 13'd0;
                        next_stage_q <= GAP_EN ? STG_GAP : STG_FIN;
                        next_state_q <= GAP_EN ? S_GAP   : S_DONE;
                        state        <= S_BARRIER;
                    end
                end

                // ------------------------------------------------------------
                // GAP: per own group, 16 lanes through select / multiply / saturate, then write; then FC
                // ------------------------------------------------------------
                S_GAP: begin
                    gap_lane  <= gap_lane + 5'd1;
                    gap_sum_r <= gap_lane_sum;
                    if (gap_lane >= 5'd1 && gap_lane <= 5'd16)
                        gap_prod <= $signed(gap_sum_r) * $signed({1'b0, GAP_INV});
                    if (gap_lane >= 5'd2)
                        gap_word[(gap_lane - 5'd2)*`DATA_W +: `DATA_W] <= gap_sat16;
                    if (gap_lane == 5'd18) begin
                        gap_lane  <= 5'd0;
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_GAP + {6'b0, ctr_g};
                        m_wr_data <= gap_word;
                        if (ctr_g_p2 >= {1'b0, CH_OUT[8:0]}) begin
                            next_stage_q <= STG_FC;
                            next_state_q <= S_MAC1;
                            state        <= S_BARRIER;
                        end else
                            ctr_g <= ctr_g_p2;
                    end
                end

                default: state <= S_IDLE;
            endcase

            if (ax_cap) h_new_r <= ax_pad_r ? {`NUM_LANES*`DATA_W{1'b0}} : m_rd_data_c;

            if (mp_wr) begin
                m_we      <= 1'b1;
                m_wr_addr <= mp_wr_addr;
                m_wr_data <= mp_acc;
            end

            // Deferred MAC group write: comb_out_r holds the result 4 cycles after the group's exit, result + bias
            // is in mac_bsum_r at 5; written from S_MAC4 at 6. The next group's bias word is read only after
            // this group's was consumed (const data lags 2 cycles).
            if (ovl_pend) begin
                ovl_cnt <= ovl_cnt + 3'd1;
                if (ovl_cnt == 3'd4) ovl_data <= comb_out_r;
                if (ovl_cnt == 3'd5 && mac_bias_en) ovl_data <= mac_bsum_r;
                if (ovl_cnt == 3'd4 && mac_bias_en) const_rd_addr <= mac_bias_base + {6'b0, ctr_g};
                if (ovl_cnt == 3'd6) ovl_pend <= 1'b0;
            end
        end
    end

    // synthesis translate_off
    // Geometry this build does not support would give silently wrong data
    reg [4:1] bar_rel_sr;
    always @(posedge clk) begin
        bar_rel_sr <= {bar_rel_sr[3:1], bar_rel};
        // Writes are counted by the barrier only once issued, so none of the old stage may follow the release. A new
        // stage writes no earlier than 6 cycles in (S_SCAN_LD ld 5), so a write in the first 4 is a late old one.
        if (!rst && |bar_rel_sr && m_we) begin
            $display("ERROR: Mamba_Cluster %0d: write to %0d within 4 cycles of a barrier release", CLUSTER_ID, m_wr_addr);
            $stop;
        end
        if ((state == S_IDLE || state == S_DONE) && start && GAP_EN && T_MAX > 10'd512) begin
            $display("ERROR: Mamba_Cluster %0d: T_MAX=%0d, the 25-bit GAP sums hold T <= 512", CLUSTER_ID, T_MAX);
            $stop;
        end
        if ((state == S_IDLE || state == S_DONE) && start && !PHASE_INC) begin
            if (ch_m_act != {CH_OUT, 1'b0}) begin
                $display("ERROR: Mamba_Cluster %0d: CH_M=%0d, CH_OUT=%0d; RN on the scan's gate row needs d_inner = 2 d_model",
                         CLUSTER_ID, CH_M, CH_OUT);
                $stop;
            end
            if (ch_m_act != 10'd8 && ch_m_act != 10'd16) begin
                $display("ERROR: Mamba_Cluster %0d: d_inner=%0d, gated RMSNorm constants exist for 128 and 256 only", CLUSTER_ID, d_inner);
                $stop;
            end
            if ((ch_m_act >> `M2_PSH) > 10'd16) begin
                $display("ERROR: Mamba_Cluster %0d: %0d heads, dt and dA words hold 16", CLUSTER_ID, ch_m_act >> `M2_PSH);
                $stop;
            end
            if (CH_OUT != 9'd4 && CH_OUT != 9'd8) begin
                $display("ERROR: Mamba_Cluster %0d: CH_OUT=%0d groups, RMSNorm and GAP support 4 or 8", CLUSTER_ID, CH_OUT);
                $stop;
            end
        end
    end
    // synthesis translate_on

endmodule
