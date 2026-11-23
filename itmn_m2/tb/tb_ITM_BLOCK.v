`timescale 1ns/1ps
`include "_parameter.v"
`include "goldens.vh"

// One ITM block, both phases, checked against Final_ITM_Full_FP (or the next block's input after MaxPool2, and
// the GAP / logits on block 4).
//   +define+ITM_BK=<0..4>   block (default 0)
//   +define+BLK_T=<N>       run only the first N timesteps (default: all)
// The weight RAMs are reloaded between the phases; the Mamba phase reads the Inception phase's P1 output in place.
// Block 0 starts from the raw 12-lead input through the encoder.
module tb_ITM_BLOCK;

`ifdef ITM_BK
    localparam BK = `ITM_BK;
`else
    localparam BK = 0;
`endif
    localparam D_IN    = 64;
    localparam D_OUT   = (BK == 4) ? 128 : 64;
    localparam D_INNER = 2 * D_OUT;
    localparam D_STATE = 16 * `M2_S;
    localparam NH      = D_INNER / (16 << `M2_PSH);
    localparam XBC_CH  = D_INNER + 2 * D_STATE;
    localparam XBC_G   = (XBC_CH + 15) / 16;
    localparam ROWS    = D_INNER + XBC_CH + NH;                  // in_proj rows [z | xBC | dt]
    localparam INP_G   = (ROWS + 15) / 16;
    localparam T_TOT   = (BK <= 1) ? 1000 : (BK <= 3) ? 500 : 250;
`ifdef BLK_T
    localparam T_RUN   = (`BLK_T < T_TOT) ? `BLK_T : T_TOT;
`else
    localparam T_RUN   = T_TOT;
`endif
    localparam CH_IN_G  = D_IN / 16;
    localparam CH_OUT_G = D_OUT / 16;
    localparam DIM      = D_OUT / 4;
    localparam DIM_G    = DIM / 16;
    localparam REACH    = 19;                       // look-ahead of the widest branch (k39)
    localparam POOL     = (BK == 1) || (BK == 3);   // MaxPool1d(2,2) after blocks 1 and 3
    localparam ENC      = (BK == 0);
    localparam GAP      = (BK == 4);
    localparam N_CLS    = 5;
    localparam [15:0] GAP_INV_V = (32768 + T_RUN/2) / T_RUN;   // round(2^15 / T)

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    reg          PHASE_INC = 1'b1;
    reg          FIN_EN    = 1'b0;
    wire         done_stage, done_all;
    reg  [9:0]   T_MAX  = T_RUN;
    reg  [8:0]   CH_OUT = CH_OUT_G;
    reg  [8:0]   CH_IN  = CH_IN_G;
    reg  [8:0]   CH_M   = D_INNER / 16;

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
        .T_MAX(T_MAX), .PHASE_INC(PHASE_INC), .CH_IN(CH_IN), .FIN_EN(FIN_EN), .POOL_OUT(POOL), .ENC_EN(ENC), .GAP_EN(GAP), .GAP_INV(GAP_INV_V),
        .CH_OUT(CH_OUT), .CH_M(CH_M),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // Goldens: activations channel-major (idx = ch*T_TOT + t), weights row-major
    reg [15:0] xin_mem  [0:D_IN*T_TOT-1];
    reg [15:0] wp1_mem  [0:D_OUT*D_IN-1];
    reg [15:0] bp1_mem  [0:D_OUT-1];
    reg [15:0] wbot_mem [0:DIM*D_OUT-1];
    reg [15:0] wb1_mem  [0:DIM*D_OUT-1];
    reg [15:0] wb2_mem  [0:DIM*DIM*9-1];
    reg [15:0] wb3_mem  [0:DIM*DIM*19-1];
    reg [15:0] wb4_mem  [0:DIM*DIM*39-1];
    reg [15:0] win_mem  [0:ROWS*D_OUT-1];
    reg [15:0] wo_mem   [0:D_OUT*D_INNER-1];
    reg [15:0] wcv_mem  [0:XBC_CH*4-1];
    reg [15:0] bcv_mem  [0:XBC_CH-1];
    reg [15:0] dtv_mem  [0:NH-1];
    reg [15:0] av_mem   [0:NH-1];
    reg [15:0] dpar_mem [0:NH-1];
    reg [15:0] gng_mem  [0:D_INNER-1];
    reg [15:0] gam_mem  [0:D_OUT-1];
    reg [15:0] isc_mem  [0:D_OUT-1];            // Inception BN scale / shift (FIN)
    reg [15:0] ish_mem  [0:D_OUT-1];
    reg [15:0] fin_exp  [0:D_OUT*T_TOT-1];
    reg [15:0] pool_exp [0:D_OUT*(T_TOT/2)-1];  // next block's input
    reg [8*40:1] bdir_next;
    reg [15:0] raw_mem  [0:12*T_TOT-1];         // [lead][t]
    reg [15:0] wenc_mem [0:64*12-1];
    reg [15:0] benc_mem [0:63];
    reg [15:0] wfc_mem  [0:N_CLS*128-1];
    reg [15:0] bfc_mem  [0:N_CLS-1];
    reg [15:0] gapg_mem [0:127];
    reg [15:0] lgg_mem  [0:N_CLS-1];
    reg signed [15:0] fin_rtl [0:D_OUT*T_TOT-1];
    reg signed [15:0] gap_e   [0:127];
    reg signed [63:0] acc64;
    integer gsum, gv, gap_errs, lg_errs, gold_errs, c;
    reg signed [15:0] gv16;

    integer g, k, t, lane, tap, ci, cyc, inc_cyc, mam_cyc, errs, shown, t_chk;
    reg [255:0] w;
    reg [15:0]  e, got;
    reg [8*40:1] bdir;

    always @(posedge clk) cyc = cyc + 1;

    // Utilization per stage id and cluster: busy cycles, cycles with a PE issue, barrier wait (booked to the
    // stage that just finished)
    integer u_cyc [0:1][0:31], u_pe [0:1][0:31], u_bar [0:1][0:31], ui, uj;
    initial for (ui = 0; ui < 2; ui = ui + 1) for (uj = 0; uj < 32; uj = uj + 1) begin
        u_cyc[ui][uj] = 0; u_pe[ui][uj] = 0; u_bar[ui][uj] = 0;
    end
    always @(posedge clk) begin
        if (dut.u_c0.state != 7'd0 && dut.u_c0.state != 7'd127 && dut.u_c0.state != 7'd1) begin
            if (dut.u_c0.state == 7'd120) u_bar[0][dut.u_c0.cur_stage] = u_bar[0][dut.u_c0.cur_stage] + 1;
            else                          u_cyc[0][dut.u_c0.cur_stage] = u_cyc[0][dut.u_c0.cur_stage] + 1;
        end
        if (dut.u_c1.state != 7'd0 && dut.u_c1.state != 7'd127 && dut.u_c1.state != 7'd1) begin
            if (dut.u_c1.state == 7'd120) u_bar[1][dut.u_c1.cur_stage] = u_bar[1][dut.u_c1.cur_stage] + 1;
            else                          u_cyc[1][dut.u_c1.cur_stage] = u_cyc[1][dut.u_c1.cur_stage] + 1;
        end
        if (dut.u_c0.cl_op_mode != `MAMBA_PE_IDLE) u_pe[0][dut.u_c0.cur_stage] = u_pe[0][dut.u_c0.cur_stage] + 1;
        if (dut.u_c1.cl_op_mode != `MAMBA_PE_IDLE) u_pe[1][dut.u_c1.cur_stage] = u_pe[1][dut.u_c1.cur_stage] + 1;
    end
    task util_report;
        begin
            $display("[UTIL] stage: c0 busy / PE issues / barrier wait | c1 busy / PE issues / barrier wait  (totals over the run)");
            for (uj = 0; uj < 32; uj = uj + 1)
                if (u_cyc[0][uj] + u_cyc[1][uj] + u_bar[0][uj] + u_bar[1][uj] != 0)
                    $display("[UTIL] stg %0d: %0d / %0d / %0d | %0d / %0d / %0d", uj,
                             u_cyc[0][uj], u_pe[0][uj], u_bar[0][uj], u_cyc[1][uj], u_pe[1][uj], u_bar[1][uj]);
        end
    endtask

    task dma_wr;
        input [2:0] target; input [14:0] addr; input [255:0] data;
        begin
            @(negedge clk);
            dma_write_en = 1; dma_target = target; dma_addr = addr; dma_wdata = data;
        end
    endtask

    task wr_both;   // both clusters' weight RAMs
        input [14:0] addr; input [255:0] data;
        begin dma_wr(3'd2, addr, data); dma_wr(3'd4, addr, data); end
    endtask

    `include "tb_wsplit.vh"

    task run_phase;
        output integer cycles;
        integer s0;
        begin
            @(negedge clk); dma_write_en = 0;
            @(negedge clk); start = 1; s0 = cyc;
            @(negedge clk); start = 0;
            wait (done_all == 1'b1);
            cycles = cyc - s0;
            repeat (4) @(negedge clk);
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
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Input_X.txt"},      xin_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Weight_Fused.txt"}, wp1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Bias_Fused.txt"},   bp1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_Bot.txt"},           wbot_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B1.txt"},            wb1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B2.txt"},            wb2_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B3.txt"},            wb3_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "W_B4.txt"},            wb4_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_InProj.txt"},    win_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_OutProj.txt"},   wo_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_Conv.txt"},      wcv_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_B_Conv.txt"},      bcv_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_DtBias.txt"},      dtv_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_A_signed.txt"},    av_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_D_param.txt"},     dpar_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_GNorm.txt"},     gng_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_Norm.txt"},      gam_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Inc_BN_Scale.txt"},    isc_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Inc_BN_Shift.txt"},    ish_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Final_ITM_Full_FP.txt"}, fin_exp);
        if (POOL) begin
            bdir_next = (BK == 1) ? "/block_02_layer03/" : "/block_04_layer06/";
            $readmemh({`GOLDEN_ROOT, bdir_next, "P1_Input_X.txt"}, pool_exp);
        end

        if (ENC) begin
            $readmemh({`GOLDEN_ROOT, "/encoder/Encoder_Input_X.txt"},      raw_mem);
            $readmemh({`GOLDEN_ROOT, "/encoder/Encoder_Weight_Fused.txt"}, wenc_mem);
            $readmemh({`GOLDEN_ROOT, "/encoder/Encoder_Bias_Fused.txt"},   benc_mem);
        end
        if (GAP) begin
            $readmemh({`GOLDEN_ROOT, "/gap_fc/FC_Weight.txt"},  wfc_mem);
            $readmemh({`GOLDEN_ROOT, "/gap_fc/FC_Bias.txt"},    bfc_mem);
            $readmemh({`GOLDEN_ROOT, "/gap_fc/GAP_Output.txt"}, gapg_mem);
            $readmemh({`GOLDEN_ROOT, "/gap_fc/FC_Logits.txt"},  lgg_mem);
        end
        repeat (4) @(negedge clk); rst = 0;
        $display("[BLK] block %0d: d_out=%0d d_inner=%0d T=%0d (running %0d)%s", BK, D_OUT, D_INNER, T_TOT, T_RUN,
                 ENC ? "  [from raw 12-lead input via encoder]" : "");

        // Inception phase
        if (ENC) begin
            // raw input one word per t (lanes 12..15 zero), encoder weight word = base + g*16 + lead
            for (t = 0; t < T_RUN; t = t + 1) begin
                w = 256'b0;
                for (lane = 0; lane < 12; lane = lane + 1) w[lane*16 +: 16] = raw_mem[lane*T_TOT + t];
                dma_wr(3'd0, `PT_ENC_IN + t, w);
            end
            for (g = 0; g < 4; g = g + 1)
                for (k = 0; k < 16; k = k + 1) begin
                    w = 256'b0;
                    if (k < 12)
                        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wenc_mem[(g*16 + lane)*12 + k];
                    w_put(2, `W_ENC_BASE, g, 16, k, w);
                end
            for (g = 0; g < 4; g = g + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = benc_mem[g*16 + lane];
                dma_wr(3'd3, `C_ENC_BIAS_BASE + g, w);
            end
        end else begin
            for (t = 0; t < T_RUN; t = t + 1)
                for (g = 0; g < CH_IN_G; g = g + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = xin_mem[(g*16 + lane)*T_TOT + t];
                    dma_wr(3'd0, `PT_X_IN + t*CH_IN_G + g, w);
                end
        end
        for (g = 0; g < CH_OUT_G; g = g + 1)
            for (k = 0; k < D_IN; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wp1_mem[(g*16 + lane)*D_IN + k];
                w_put(2, `W_P1_BASE, g, D_IN, k, w);
            end
        for (g = 0; g < DIM_G; g = g + 1)
            for (k = 0; k < D_OUT; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wbot_mem[(g*16 + lane)*D_OUT + k];
                w_put(2, `W_BOT_BASE, g, D_OUT, k, w);
                for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wb1_mem[(g*16 + lane)*D_OUT + k];
                w_put(2, `W_B1_BASE, g, D_OUT, k, w);
            end
        for (g = 0; g < DIM_G; g = g + 1)
            for (tap = 0; tap < 39; tap = tap + 1)
                for (ci = 0; ci < DIM; ci = ci + 1) begin
                    if (tap < 9) begin
                        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wb2_mem[((g*16 + lane)*DIM + ci)*9 + tap];
                        w_put(2, `W_B2_BASE, g, 9*DIM, tap*DIM + ci, w);
                    end
                    if (tap < 19) begin
                        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wb3_mem[((g*16 + lane)*DIM + ci)*19 + tap];
                        w_put(2, `W_B3_BASE, g, 19*DIM, tap*DIM + ci, w);
                    end
                    for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wb4_mem[((g*16 + lane)*DIM + ci)*39 + tap];
                    w_put(2, `W_B4_BASE, g, 39*DIM, tap*DIM + ci, w);
                end
        for (g = 0; g < CH_OUT_G; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = bp1_mem[g*16 + lane];
            dma_wr(3'd3, `C_P1_BIAS_BASE + g, w);
        end
        PHASE_INC = 1'b1; FIN_EN = 1'b0;
        wx_flush;
        run_phase(inc_cyc);
        $display("[BLK] INC phase   = %0d cycles (%0d per t)", inc_cyc, inc_cyc / T_RUN);

        // Mamba phase
        for (g = 0; g < INP_G; g = g + 1)
            for (k = 0; k < D_OUT; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = ((g*16 + lane) < ROWS) ? win_mem[(g*16 + lane)*D_OUT + k] : 16'h0000;
                w_put(g % 2, `W_INPROJ_BASE, g / 2, D_OUT, k, w);
            end
        for (g = 0; g < D_OUT/16; g = g + 1)
            for (k = 0; k < D_INNER; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wo_mem[(g*16 + lane)*D_INNER + k];
                w_put(g % 2, `W_OUTPROJ_BASE, g / 2, D_INNER, k, w);
            end
        for (g = 0; g < XBC_G; g = g + 1) begin
            for (k = 0; k < 4; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wcv_mem[(g*16 + lane)*4 + k];
                wr_both(`W_CONV_BASE + g*4 + k, w);
            end
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = bcv_mem[g*16 + lane];
            dma_wr(3'd3, `C_B_CONV_BASE + g, w);
        end
        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = (lane < NH) ? dtv_mem[lane] : 16'h0000;
        dma_wr(3'd3, `C_DT_BIAS_BASE, w);
        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = (lane < NH) ? av_mem[lane] : 16'h0000;
        dma_wr(3'd3, `C_A_BASE, w);
        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = (lane < NH) ? dpar_mem[lane] : 16'h0000;
        dma_wr(3'd3, `C_D_BASE, w);
        for (g = 0; g < D_INNER/16; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = gng_mem[g*16 + lane];
            dma_wr(3'd3, `C_W_GNORM_BASE + g, w);
        end
        for (g = 0; g < D_OUT/16; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = gam_mem[g*16 + lane];
            dma_wr(3'd3, `C_W_NORM_BASE + g, w);
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = isc_mem[g*16 + lane];
            dma_wr(3'd3, `C_INC_SCALE_BASE + g, w);
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = ish_mem[g*16 + lane];
            dma_wr(3'd3, `C_INC_SHIFT_BASE + g, w);
        end
        if (GAP) begin
            // FC: one output group (lane = class) in cluster 0's weights
            for (k = 0; k < 128; k = k + 1) begin
                w = 256'b0;
                for (lane = 0; lane < N_CLS; lane = lane + 1) w[lane*16 +: 16] = wfc_mem[lane*128 + k];
                w_put(0, `W_FC_BASE, 0, 128, k, w);
            end
            w = 256'b0;
            for (lane = 0; lane < N_CLS; lane = lane + 1) w[lane*16 +: 16] = bfc_mem[lane];
            dma_wr(3'd3, `C_FC_BIAS_BASE, w);
        end
        PHASE_INC = 1'b0; FIN_EN = 1'b1;
        wx_flush;
        run_phase(mam_cyc);
        $display("[BLK] Mamba phase = %0d cycles (%0d per t)", mam_cyc, mam_cyc / T_RUN);
        util_report;

        // A truncated run cannot match the full-T golden in its last REACH timesteps
        t_chk = (T_RUN < T_TOT) ? T_RUN - REACH : T_RUN;
        if (POOL) t_chk = t_chk / 2;
        errs = 0; shown = 0;
        for (t = 0; t < t_chk; t = t + 1)
            for (g = 0; g < CH_OUT_G; g = g + 1) begin
                @(negedge clk);
                dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_INC_OUT + t*CH_OUT_G + g;
                @(negedge clk); @(negedge clk);
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    e   = POOL ? pool_exp[(g*16 + lane)*(T_TOT/2) + t] : fin_exp[(g*16 + lane)*T_TOT + t];
                    got = dma_rdata[lane*16 +: 16];
                    if (got !== e || (^got === 1'bx) || (^e === 1'bx)) begin   // x never counts as a match
                        errs = errs + 1;
                        if (shown < 8) begin
                            $display("  %s MISMATCH t=%0d c=%0d  got=%04h exp=%04h", POOL ? "POOLED" : "FINAL", t, g*16 + lane, got, e);
                            shown = shown + 1;
                        end
                    end
                end
            end
        dma_read_en = 0;
        if (POOL) $display("  POOLED errors: %0d / %0d", errs, t_chk * D_OUT);
        else      $display("  FINAL errors: %0d / %0d", errs, t_chk * D_OUT);
        if (GAP) check_head;
        if (errs == 0) $display(">>>>> ITM BLOCK %0d: BYTE-EXACT PASS <<<<<", BK);
        else           $display(">>>>> ITM BLOCK %0d: FAIL <<<<<", BK);
        $finish;
    end

    // GAP and logits recomputed from the RTL's block output (valid for truncated runs too); with the full T also
    // compared with the goldens
    task check_head;
        begin
            for (t = 0; t < T_RUN; t = t + 1)
                for (g = 0; g < CH_OUT_G; g = g + 1) begin
                    @(negedge clk);
                    dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_INC_OUT + t*CH_OUT_G + g;
                    @(negedge clk); @(negedge clk);
                    for (lane = 0; lane < 16; lane = lane + 1)
                        fin_rtl[(g*16 + lane)*T_TOT + t] = dma_rdata[lane*16 +: 16];
                end
            gap_errs = 0; gold_errs = 0;
            for (g = 0; g < CH_OUT_G; g = g + 1) begin
                @(negedge clk);
                dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_GAP + g;
                @(negedge clk); @(negedge clk);
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    c = g*16 + lane;
                    gsum = 0;
                    for (t = 0; t < T_RUN; t = t + 1) gsum = gsum + fin_rtl[c*T_TOT + t];
                    gv = (gsum * $signed({1'b0, GAP_INV_V})) >>> 15;
                    gap_e[c] = (gv > 32767) ? 16'sh7FFF : (gv < -32768) ? 16'sh8000 : gv[15:0];
                    gv16 = dma_rdata[lane*16 +: 16];
                    if (gv16 !== gap_e[c]) begin
                        gap_errs = gap_errs + 1;
                        if (gap_errs <= 4) $display("  GAP MISMATCH c=%0d got=%04h exp=%04h", c, gv16, gap_e[c]);
                    end
                    if (T_RUN == T_TOT && gv16 !== gapg_mem[c]) gold_errs = gold_errs + 1;
                end
            end
            $display("  GAP errors: %0d / %0d", gap_errs, D_OUT);
            lg_errs = 0;
            @(negedge clk);
            dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_LOGITS;
            @(negedge clk); @(negedge clk);
            for (lane = 0; lane < N_CLS; lane = lane + 1) begin
                acc64 = 0;
                for (c = 0; c < 128; c = c + 1)
                    acc64 = acc64 + $signed(wfc_mem[lane*128 + c]) * gap_e[c];
                acc64 = acc64 >>> 11;
                if (acc64 > 32767) acc64 = 32767; else if (acc64 < -32768) acc64 = -32768;
                acc64 = acc64 + $signed(bfc_mem[lane]);
                if (acc64 > 32767) acc64 = 32767; else if (acc64 < -32768) acc64 = -32768;
                gv16 = dma_rdata[lane*16 +: 16];
                $display("  logit[%0d] = %6d (exp %6d%s)", lane, gv16, $signed(acc64[15:0]),
                         (T_RUN == T_TOT) ? "" : ", from RTL FINAL");
                if (gv16 !== acc64[15:0]) lg_errs = lg_errs + 1;
                if (T_RUN == T_TOT && gv16 !== $signed(lgg_mem[lane])) gold_errs = gold_errs + 1;
            end
            dma_read_en = 0;
            $display("  LOGITS errors: %0d / %0d", lg_errs, N_CLS);
            if (T_RUN == T_TOT) $display("  HEAD vs golden GAP_Output/FC_Logits errors: %0d / %0d", gold_errs, D_OUT + N_CLS);
            errs = errs + gap_errs + lg_errs + gold_errs;
        end
    endtask

    initial begin
        #2_000_000_000;
        $display("[BLK] TIMEOUT");
        $finish;
    end
endmodule
