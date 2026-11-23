`timescale 1ns/1ps
`include "_parameter.v"
`include "goldens.vh"

// End to end: raw 12-lead ECG -> 5 ITM blocks -> GAP -> FC. The host loads only the raw input, weights and
// constants; block b's output stays in PT_INC_OUT as block b+1's input. Every block's output is checked (to
// localise a failure), the run passes when the logits match.
//   +define+E2E_LAST=<0..4>  stop after this block (default 4)
module tb_ITM_E2E;

`ifdef E2E_LAST
    localparam LAST = `E2E_LAST;
`else
    localparam LAST = 4;
`endif
    localparam N_CLS = 5;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    reg          PHASE_INC = 0, FIN_EN = 0, POOL_OUT = 0, ENC_EN = 0, GAP_EN = 0;
    wire         done_stage, done_all;
    reg  [9:0]   T_MAX;
    reg  [8:0]   CH_OUT, CH_IN, CH_M;
    reg  [15:0]  GAP_INV = 16'd131;            // round(2^15 / 250)

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
        .T_MAX(T_MAX), .PHASE_INC(PHASE_INC), .CH_IN(CH_IN), .FIN_EN(FIN_EN),
        .POOL_OUT(POOL_OUT), .ENC_EN(ENC_EN), .GAP_EN(GAP_EN), .GAP_INV(GAP_INV),
        .CH_OUT(CH_OUT), .CH_M(CH_M),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // Goldens, sized for the largest block
    reg [15:0] raw_mem  [0:12*1000-1];
    reg [15:0] wenc_mem [0:64*12-1];
    reg [15:0] benc_mem [0:63];
    reg [15:0] wp1_mem  [0:128*64-1];
    reg [15:0] bp1_mem  [0:127];
    reg [15:0] wbot_mem [0:32*128-1];
    reg [15:0] wb1_mem  [0:32*128-1];
    reg [15:0] wb2_mem  [0:32*32*9-1];
    reg [15:0] wb3_mem  [0:32*32*19-1];
    reg [15:0] wb4_mem  [0:32*32*39-1];
    reg [15:0] win_mem  [0:(2*256 + 32*`M2_S + 16)*128-1];
    reg [15:0] wo_mem   [0:128*256-1];
    reg [15:0] wcv_mem  [0:(256 + 32*`M2_S)*4-1];
    reg [15:0] bcv_mem  [0:256 + 32*`M2_S - 1];
    reg [15:0] dtv_mem  [0:15];
    reg [15:0] av_mem   [0:15];
    reg [15:0] dpar_mem [0:15];
    reg [15:0] gng_mem  [0:255];
    reg [15:0] gam_mem  [0:127];
    reg [15:0] isc_mem  [0:127];
    reg [15:0] ish_mem  [0:127];
    reg [15:0] out_exp  [0:64000-1];      // block output, or the next block's input after MaxPool2
    reg [15:0] wfc_mem  [0:N_CLS*128-1];
    reg [15:0] bfc_mem  [0:N_CLS-1];
    reg [15:0] lgg_mem  [0:N_CLS-1];
    reg [15:0] gapg_mem [0:127];

    integer bk, D_OUT, D_INNER, NH, ROWS, INP_G, XBC_G, T, CH_OUT_G, DIM, DIM_G, T_OUT, pool;
    integer g, k, t, lane, tap, ci, cyc, ph_cyc, errs, shown, total_errs;
    integer inc_cyc [0:4];
    integer mam_cyc [0:4];
    reg [255:0] w;
    reg [15:0]  e, got;
    reg [8*40:1] bdir, bdir_next;

    always @(posedge clk) cyc = cyc + 1;

    task dma_wr;
        input [2:0] target; input [14:0] addr; input [255:0] data;
        begin
            @(negedge clk);
            dma_write_en = 1; dma_target = target; dma_addr = addr; dma_wdata = data;
        end
    endtask
    task wr_both;
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

    task set_block;
        input integer b;
        begin
            case (b)
                0: begin bdir = "/block_00_layer00/"; T = 1000; end
                1: begin bdir = "/block_01_layer01/"; T = 1000; end
                2: begin bdir = "/block_02_layer03/"; T = 500;  end
                3: begin bdir = "/block_03_layer04/"; T = 500;  end
                default: begin bdir = "/block_04_layer06/"; T = 250; end
            endcase
            D_OUT    = (b == 4) ? 128 : 64;
            D_INNER  = 2 * D_OUT;
            NH       = D_INNER / (16 << `M2_PSH);
            ROWS     = D_INNER + (D_INNER + 32 * `M2_S) + NH;
            INP_G    = (ROWS + 15) / 16;
            XBC_G    = (D_INNER + 32 * `M2_S + 15) / 16;
            CH_OUT_G = D_OUT / 16;
            DIM      = D_OUT / 4;
            DIM_G    = DIM / 16;
            pool     = (b == 1) || (b == 3);
            T_OUT    = pool ? T / 2 : T;
            T_MAX    = T;
            CH_OUT   = CH_OUT_G;
            CH_IN    = 4;
            CH_M     = D_INNER / 16;
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
            if (pool) begin
                bdir_next = (b == 1) ? "/block_02_layer03/" : "/block_04_layer06/";
                $readmemh({`GOLDEN_ROOT, bdir_next, "P1_Input_X.txt"}, out_exp);
            end else begin
                $readmemh({`GOLDEN_ROOT, bdir, "Final_ITM_Full_FP.txt"}, out_exp);
            end
        end
    endtask

    task load_inc;     // Inception weights, P1 bias; block 0 also the raw input and the encoder
        begin
            if (bk == 0) begin
                for (t = 0; t < T; t = t + 1) begin
                    w = 256'b0;
                    for (lane = 0; lane < 12; lane = lane + 1) w[lane*16 +: 16] = raw_mem[lane*1000 + t];
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
            end
            for (g = 0; g < CH_OUT_G; g = g + 1)
                for (k = 0; k < 64; k = k + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = wp1_mem[(g*16 + lane)*64 + k];
                    w_put(2, `W_P1_BASE, g, 64, k, w);
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
        end
    endtask

    task load_mamba;   // Mamba weights and constants, Inception BN; block 4 also the FC head
        begin
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
            if (bk == 4) begin
                for (k = 0; k < 128; k = k + 1) begin
                    w = 256'b0;
                    for (lane = 0; lane < N_CLS; lane = lane + 1) w[lane*16 +: 16] = wfc_mem[lane*128 + k];
                    w_put(0, `W_FC_BASE, 0, 128, k, w);
                end
                w = 256'b0;
                for (lane = 0; lane < N_CLS; lane = lane + 1) w[lane*16 +: 16] = bfc_mem[lane];
                dma_wr(3'd3, `C_FC_BIAS_BASE, w);
            end
        end
    endtask

    task check_block_out;
        begin
            errs = 0; shown = 0;
            for (t = 0; t < T_OUT; t = t + 1)
                for (g = 0; g < CH_OUT_G; g = g + 1) begin
                    @(negedge clk);
                    dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_INC_OUT + t*CH_OUT_G + g;
                    @(negedge clk); @(negedge clk);
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        e   = out_exp[(g*16 + lane)*T_OUT + t];
                        got = dma_rdata[lane*16 +: 16];
                        if (got !== e || (^got === 1'bx) || (^e === 1'bx)) begin   // x never counts as a match
                            errs = errs + 1;
                            if (shown < 6) begin
                                $display("  B%0d OUT MISMATCH t=%0d c=%0d got=%04h exp=%04h", bk, t, g*16 + lane, got, e);
                                shown = shown + 1;
                            end
                        end
                    end
                end
            dma_read_en = 0;
            $display("  B%0d %s errors: %0d / %0d", bk, pool ? "POOLED" : "FINAL", errs, T_OUT * D_OUT);
            total_errs = total_errs + errs;
        end
    endtask

    initial begin
        cyc = 0; total_errs = 0;
        $readmemh({`GOLDEN_ROOT, "/encoder/Encoder_Input_X.txt"},      raw_mem);
        $readmemh({`GOLDEN_ROOT, "/encoder/Encoder_Weight_Fused.txt"}, wenc_mem);
        $readmemh({`GOLDEN_ROOT, "/encoder/Encoder_Bias_Fused.txt"},   benc_mem);
        $readmemh({`GOLDEN_ROOT, "/gap_fc/FC_Weight.txt"},             wfc_mem);
        $readmemh({`GOLDEN_ROOT, "/gap_fc/FC_Bias.txt"},               bfc_mem);
        $readmemh({`GOLDEN_ROOT, "/gap_fc/FC_Logits.txt"},             lgg_mem);
        $readmemh({`GOLDEN_ROOT, "/gap_fc/GAP_Output.txt"},            gapg_mem);
        repeat (4) @(negedge clk); rst = 0;
        $display("[E2E] raw 12-lead ECG -> blocks 0..%0d%s", LAST, (LAST == 4) ? " -> GAP -> FC" : "");

        for (bk = 0; bk <= LAST; bk = bk + 1) begin
            set_block(bk);
            load_inc;
            PHASE_INC = 1; FIN_EN = 0; POOL_OUT = 0; GAP_EN = 0; ENC_EN = (bk == 0);
            wx_flush;
            run_phase(ph_cyc); inc_cyc[bk] = ph_cyc;
            load_mamba;
            PHASE_INC = 0; FIN_EN = 1; POOL_OUT = pool; GAP_EN = (bk == 4); ENC_EN = 0;
            wx_flush;
            run_phase(ph_cyc); mam_cyc[bk] = ph_cyc;
            $display("[E2E] block %0d: INC %0d + Mamba %0d = %0d cycles (T=%0d)",
                     bk, inc_cyc[bk], mam_cyc[bk], inc_cyc[bk] + mam_cyc[bk], T);
            check_block_out;
        end

        if (LAST == 4) begin
            errs = 0;
            for (g = 0; g < 8; g = g + 1) begin
                @(negedge clk);
                dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_GAP + g;
                @(negedge clk); @(negedge clk);
                for (lane = 0; lane < 16; lane = lane + 1)
                    if (dma_rdata[lane*16 +: 16] !== gapg_mem[g*16 + lane] || (^dma_rdata[lane*16 +: 16] === 1'bx) || (^gapg_mem[g*16 + lane] === 1'bx)) errs = errs + 1;
            end
            $display("  GAP errors: %0d / 128", errs);
            total_errs = total_errs + errs;
            errs = 0;
            @(negedge clk);
            dma_read_en = 1; dma_rtarget = 3'd0; dma_raddr = `PT_LOGITS;
            @(negedge clk); @(negedge clk);
            for (lane = 0; lane < N_CLS; lane = lane + 1) begin
                got = dma_rdata[lane*16 +: 16];
                $display("  logit[%0d] = %6d   golden %6d", lane, $signed(got), $signed(lgg_mem[lane]));
                if (got !== lgg_mem[lane] || (^got === 1'bx) || (^lgg_mem[lane] === 1'bx)) errs = errs + 1;
            end
            dma_read_en = 0;
            $display("  LOGITS errors: %0d / %0d", errs, N_CLS);
            total_errs = total_errs + errs;
        end

        ph_cyc = 0;
        for (bk = 0; bk <= LAST; bk = bk + 1) ph_cyc = ph_cyc + inc_cyc[bk] + mam_cyc[bk];
        $display("[E2E] total compute = %0d cycles (%0d.%02d ms at 250 MHz)", ph_cyc,
                 (ph_cyc + 1250) / 250000, ((ph_cyc + 1250) / 2500) % 100);
        if (total_errs == 0) $display(">>>>> E2E BYTE-EXACT PASS: raw ECG -> FC_Logits <<<<<");
        else                 $display(">>>>> E2E FAIL: %0d mismatches <<<<<", total_errs);
        $finish;
    end
endmodule
