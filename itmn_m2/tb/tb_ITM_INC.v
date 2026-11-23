`timescale 1ns/1ps
`include "_parameter.v"
`include "goldens.vh"

// Inception phase of one block; the P1, B1, BOT and B2..B4 outputs are checked against the goldens.
//   +define+ITM_BK=<0..4>   block (default 0)
//   +define+INC_T=<N>       run only the first N timesteps (default: all)
// Time-split stages need all output groups in both clusters, so every weight goes into both weight RAMs.
module tb_ITM_INC;

`ifdef ITM_BK
    localparam BK = `ITM_BK;
`else
    localparam BK = 0;
`endif

    // d_in is 64 for every block
    localparam D_IN   = 64;
    localparam D_OUT  = (BK == 4) ? 128 : 64;
    localparam T_TOT  = (BK <= 1) ? 1000 : (BK <= 3) ? 500 : 250;
`ifdef INC_T
    localparam T_RUN  = (`INC_T < T_TOT) ? `INC_T : T_TOT;
`else
    localparam T_RUN  = T_TOT;
`endif
    localparam CH_IN_G  = D_IN / 16;
    localparam CH_OUT_G = D_OUT / 16;
    localparam DIM      = D_OUT / 4;          // branch width
    localparam DIM_G    = DIM / 16;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    wire         done_stage, done_all;
    reg  [9:0]   T_MAX         = T_RUN;
    reg  [8:0]   CH_OUT        = CH_OUT_G;
    reg  [8:0]   CH_IN         = CH_IN_G;
    reg  [8:0]   CH_M          = 2 * CH_OUT_G;

    reg          dma_write_en = 0;
    reg  [2:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    reg          dma_read_en  = 0;
    reg  [2:0]   dma_rtarget  = 0;
    reg  [14:0]  dma_raddr    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top dut (
        .clk(clk), .rst(rst), .start(start),
        .done_stage(done_stage), .done_all(done_all),
        .T_MAX(T_MAX), .PHASE_INC(1'b1), .CH_IN(CH_IN), .FIN_EN(1'b0), .POOL_OUT(1'b0), .ENC_EN(1'b0), .GAP_EN(1'b0), .GAP_INV(16'd0),
        .CH_OUT(CH_OUT), .CH_M(CH_M),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // Goldens: activations channel-major (idx = ch*T_TOT + t)
    reg [15:0] xin_mem [0:D_IN*T_TOT-1];
    reg [15:0] wp1_mem [0:D_OUT*D_IN-1];       // [co][ci]
    reg [15:0] bp1_mem [0:D_OUT-1];
    reg [15:0] p1_exp  [0:D_OUT*T_TOT-1];
    reg [15:0] wbot_mem [0:DIM*D_OUT-1];
    reg [15:0] wb1_mem  [0:DIM*D_OUT-1];
    reg [15:0] bot_exp  [0:DIM*T_TOT-1];
    reg [15:0] b1_exp   [0:DIM*T_TOT-1];
    reg [15:0] wb2_mem  [0:DIM*DIM*9-1];      // [co][ci][tap]
    reg [15:0] wb3_mem  [0:DIM*DIM*19-1];
    reg [15:0] wb4_mem  [0:DIM*DIM*39-1];
    reg [15:0] b2_exp   [0:DIM*T_TOT-1];
    reg [15:0] b3_exp   [0:DIM*T_TOT-1];
    reg [15:0] b4_exp   [0:DIM*T_TOT-1];
    integer g, k, t, lane, cyc0, cyc1, cyc, total_errs;
    reg [255:0] w;
    reg [8*40:1] bdir;

    always @(posedge clk) cyc = cyc + 1;

    task dma_wr;
        input [2:0] target; input [14:0] addr; input [255:0] data;
        begin
            @(negedge clk);
            dma_write_en = 1; dma_target = target; dma_addr = addr; dma_wdata = data;
        end
    endtask

    `include "tb_wsplit.vh"

    task dma_rd;
        input [14:0] addr;
        begin
            @(negedge clk);
            dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = addr;
            @(negedge clk); @(negedge clk);
        end
    endtask

    initial begin
        cyc = 0;
        case (BK)
            0: bdir = "/block_00_layer00/";
            1: bdir = "/block_01_layer01/";
            2: bdir = "/block_02_layer03/";
            3: bdir = "/block_03_layer04/";
            default: bdir = "/block_04_layer06/";
        endcase
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Input_X.txt"},          xin_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Weight_Fused.txt"},     wp1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Bias_Fused.txt"},       bp1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Output_Golden_FP.txt"}, p1_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "W_Bot.txt"},               wbot_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B1.txt"},                wb1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Out_Bot.txt"},             bot_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Out_B1.txt"},              b1_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B2.txt"},                wb2_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B3.txt"},                wb3_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B4.txt"},                wb4_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Out_B2.txt"},              b2_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Out_B3.txt"},              b3_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Out_B4.txt"},              b4_exp);

        repeat (4) @(negedge clk); rst = 0;
        $display("[INC] block %0d: d_in=%0d d_out=%0d T=%0d (checking %0d)", BK, D_IN, D_OUT, T_TOT, T_RUN);

        for (t = 0; t < T_RUN; t = t + 1)
            for (g = 0; g < CH_IN_G; g = g + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = xin_mem[(g*16 + lane)*T_TOT + t];
                dma_wr(3'd0, `PT_X_IN + t*CH_IN_G + g, w);
            end
        // weight word = base + g*len + k
        for (g = 0; g < CH_OUT_G; g = g + 1)
            for (k = 0; k < D_IN; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = wp1_mem[(g*16 + lane)*D_IN + k];
                w_put(2, `W_P1_BASE, g, D_IN, k, w);
            end
        for (g = 0; g < DIM_G; g = g + 1)
            for (k = 0; k < D_OUT; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = wbot_mem[(g*16 + lane)*D_OUT + k];
                w_put(2, `W_BOT_BASE, g, D_OUT, k, w);
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = wb1_mem[(g*16 + lane)*D_OUT + k];
                w_put(2, `W_B1_BASE, g, D_OUT, k, w);
            end
        // K-tap conv weight word = base + g*K*DIM + tap*DIM + ci
        load_conv(`W_B2_BASE, 9);
        load_conv(`W_B3_BASE, 19);
        load_conv(`W_B4_BASE, 39);
        for (g = 0; g < CH_OUT_G; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1)
                w[lane*16 +: 16] = bp1_mem[g*16 + lane];
            dma_wr(3'd3, `C_P1_BIAS_BASE + g, w);
        end
        wx_flush;
        @(negedge clk); dma_write_en = 0;

        @(negedge clk); start = 1; cyc0 = cyc;
        @(negedge clk); start = 0;
        wait (done_all == 1'b1);
        cyc1 = cyc;
        repeat (4) @(negedge clk);
        $display("[INC] compute = %0d cycles (%0d per t)", cyc1 - cyc0, (cyc1 - cyc0) / T_RUN);

        total_errs = 0;
        // last argument: how far the stage looks ahead in t; a truncated run cannot match the full-T golden
        // in its last `reach` timesteps, so they are skipped
        check_region(0, `PT_INPUT,             CH_OUT_G, CH_OUT_G, 0);    // P1
        check_region(1, `PT_INC_OUT,           CH_OUT_G, DIM_G,    1);    // B1 (maxpool3)
        check_region(2, `PT_BOT,               DIM_G,    DIM_G,    0);    // BOT
        check_region(3, `PT_INC_OUT + 1*DIM_G, CH_OUT_G, DIM_G,    4);    // B2 (k9)
        check_region(4, `PT_INC_OUT + 2*DIM_G, CH_OUT_G, DIM_G,    9);    // B3 (k19)
        check_region(5, `PT_INC_OUT + 3*DIM_G, CH_OUT_G, DIM_G,    19);   // B4 (k39)
        if (total_errs == 0) $display(">>>>> INC BLOCK %0d: BYTE-EXACT PASS (P1, B1, BOT, B2, B3, B4) <<<<<", BK);
        else                 $display(">>>>> INC BLOCK %0d: FAIL (%0d mismatches) <<<<<", BK, total_errs);
        $finish;
    end

    task load_conv;
        input integer base; input integer K;
        integer g_, t_, c_, l_;
        reg [255:0] w_;
        begin
            for (g_ = 0; g_ < DIM_G; g_ = g_ + 1)
                for (t_ = 0; t_ < K; t_ = t_ + 1)
                    for (c_ = 0; c_ < DIM; c_ = c_ + 1) begin
                        for (l_ = 0; l_ < 16; l_ = l_ + 1)
                            case (K)
                                9:  w_[l_*16 +: 16] = wb2_mem[((g_*16 + l_)*DIM + c_)*9  + t_];
                                19: w_[l_*16 +: 16] = wb3_mem[((g_*16 + l_)*DIM + c_)*19 + t_];
                                default: w_[l_*16 +: 16] = wb4_mem[((g_*16 + l_)*DIM + c_)*39 + t_];
                            endcase
                        w_put(2, base, g_, K*DIM, t_*DIM + c_, w_);
                    end
        end
    endtask

    // which: 0 P1, 1 B1, 2 BOT, 3..5 B2..B4; region word = base + t*wpt + g, g < ngrp
    task check_region;
        input integer which; input integer base; input integer wpt; input integer ngrp;
        input integer reach;
        integer t_, g_, l_, errs_, shown_, nch, t_chk;
        reg [15:0] e_, got_;
        begin
            errs_ = 0; shown_ = 0; nch = ngrp * 16;
            t_chk = (T_RUN < T_TOT) ? T_RUN - reach : T_RUN;
            for (t_ = 0; t_ < t_chk; t_ = t_ + 1)
                for (g_ = 0; g_ < ngrp; g_ = g_ + 1) begin
                    dma_rd(base + t_*wpt + g_);
                    for (l_ = 0; l_ < 16; l_ = l_ + 1) begin
                        case (which)
                            0: e_ = p1_exp [(g_*16 + l_)*T_TOT + t_];
                            1: e_ = b1_exp [(g_*16 + l_)*T_TOT + t_];
                            3: e_ = b2_exp [(g_*16 + l_)*T_TOT + t_];
                            4: e_ = b3_exp [(g_*16 + l_)*T_TOT + t_];
                            5: e_ = b4_exp [(g_*16 + l_)*T_TOT + t_];
                            default: e_ = bot_exp[(g_*16 + l_)*T_TOT + t_];
                        endcase
                        got_ = dma_rdata[l_*16 +: 16];
                        if (got_ !== e_ || (^got_ === 1'bx)) begin
                            errs_ = errs_ + 1;
                            if (shown_ < 6) begin
                                $display("  %s MISMATCH t=%0d c=%0d  got=%04h exp=%04h",
                                         (which == 0) ? "P1 " : (which == 1) ? "B1 " : (which == 2) ? "BOT" : (which == 3) ? "B2 " : (which == 4) ? "B3 " : "B4 ", t_, g_*16 + l_, got_, e_);
                                shown_ = shown_ + 1;
                            end
                        end
                    end
                end
            dma_read_en = 0;
            $display("  %s errors: %0d / %0d", (which == 0) ? "P1" : (which == 1) ? "B1" : (which == 2) ? "BOT" : (which == 3) ? "B2" : (which == 4) ? "B3" : "B4", errs_, t_chk * nch);
            total_errs = total_errs + errs_;
        end
    endtask

    initial begin
        #400_000_000;
        $display("[INC] TIMEOUT");
        $finish;
    end
endmodule
