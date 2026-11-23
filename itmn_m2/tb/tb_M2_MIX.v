`timescale 1ns/1ps
`include "_parameter.v"
`include "goldens.vh"

// Mamba2 mixer of one block (Mamba phase only), every intermediate checked against the goldens.
//   +define+ITM_BK=<0..4>   block (default 0)
//   +define+MIX_T=<N>       timesteps to run (default 4)
// in_proj frames are circular (4 deep), so they are checked for the last min(N, 4) timesteps; the single-slot
// regions for the last timestep; out_proj for every timestep; the scan state only when the whole T is run. The
// write monitor (m2_write_mon.vh) checks every write of both clusters at its own timestep.
module tb_M2_MIX;

`ifdef ITM_BK
    localparam BK = `ITM_BK;
`else
    localparam BK = 0;
`endif

    localparam D_MODEL = (BK == 4) ? 128 : 64;
    localparam D_INNER = 2 * D_MODEL;
    localparam S_CT    = `M2_S;
    localparam NS      = 16 * S_CT;                              // d_state
    localparam HD      = 16 << `M2_PSH;                          // headdim
    localparam NH      = D_INNER / HD;
    localparam XBC_CH  = D_INNER + 2 * NS;
    localparam ROWS    = D_INNER + XBC_CH + NH;                 // in_proj output rows
    localparam T_TOT   = (BK <= 1) ? 1000 : (BK <= 3) ? 500 : 250;
`ifdef MIX_T
    localparam T_RUN   = (`MIX_T < T_TOT) ? `MIX_T : T_TOT;
`else
    localparam T_RUN   = 4;
`endif
    localparam DM_G    = D_MODEL / 16;
    localparam DI_G    = D_INNER / 16;
    localparam XBC_G   = (XBC_CH + 15) / 16;
    localparam INP_G   = (ROWS + 15) / 16;
    localparam NCHK    = (T_RUN < 4) ? T_RUN : 4;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    wire         done_stage, done_all;
    reg  [9:0]   T_MAX  = T_RUN;
    reg  [8:0]   CH_OUT = DM_G;
    reg  [8:0]   CH_M   = DI_G;
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
        .T_MAX(T_MAX), .CH_OUT(CH_OUT), .CH_M(CH_M), .PHASE_INC(1'b0), .CH_IN(CH_OUT),
        .FIN_EN(1'b0), .POOL_OUT(1'b0), .ENC_EN(1'b0), .GAP_EN(1'b0), .GAP_INV(16'd0),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // Goldens: activations channel-major (idx = ch*T_TOT + t), weights row-major
    reg [15:0] p1_mem   [0:D_MODEL*T_TOT-1];
    reg [15:0] gam_mem  [0:D_MODEL-1];
    reg [15:0] win_mem  [0:ROWS*D_MODEL-1];
    reg [15:0] xn_exp   [0:D_MODEL*T_TOT-1];
    reg [15:0] inp_exp  [0:ROWS*T_TOT-1];
    reg [15:0] wcv_mem  [0:XBC_CH*4-1];                // [ch][tap]
    reg [15:0] bcv_mem  [0:XBC_CH-1];
    reg [15:0] cv_exp   [0:XBC_CH*T_TOT-1];
    reg [15:0] xbc_exp  [0:XBC_CH*T_TOT-1];
    reg [15:0] dtv_mem  [0:NH-1];
    reg [15:0] av_mem   [0:NH-1];
    reg [15:0] dt_exp   [0:NH*T_TOT-1];
    reg [15:0] da_exp   [0:NH*T_TOT-1];
    reg [15:0] dpar_mem [0:NH-1];
    reg [15:0] yssm_exp [0:D_INNER*T_TOT-1];
    reg [15:0] hst_exp  [0:D_INNER*NS-1];              // state after the last timestep, [channel][n]
    reg [15:0] gng_mem  [0:D_INNER-1];
    reg [15:0] yg_exp   [0:D_INNER*T_TOT-1];
    reg [15:0] yn_exp   [0:D_INNER*T_TOT-1];
    reg [15:0] wo_mem   [0:D_MODEL*D_INNER-1];
    reg [15:0] op_exp   [0:D_MODEL*T_TOT-1];

    integer g, k, t, tt, lane, cyc, cyc0, cyc1, errs, total_errs, hc, hl, hs;
    reg [255:0] w, hword;
    reg [8*40:1] bdir;

    always @(posedge clk) cyc = cyc + 1;

    // Cycles of cluster 0 per stage id; the one-off H_INIT and the barriers separately
    integer prof [0:31]; integer pi, prof_bar, prof_hinit;
    initial begin for (pi = 0; pi < 32; pi = pi + 1) prof[pi] = 0; prof_bar = 0; prof_hinit = 0; end
    always @(posedge clk) if (dut.u_c0.state != 7'd127 && dut.u_c0.state != 7'd0 && start == 0) begin
        if (dut.u_c0.state == 7'd120)    prof_bar   = prof_bar + 1;
        else if (dut.u_c0.state == 7'd1) prof_hinit = prof_hinit + 1;
        else prof[dut.u_c0.cur_stage] = prof[dut.u_c0.cur_stage] + 1;
    end

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

    // Compare the word on dma_rdata with golden `which` (0 X_NORM, 1 in_proj, 2 conv, 3 xBC, 4 dt, 5 dA, 6 y_ssm,
    // 7 y_gated, 8 y_norm, 9 out_proj) at group grp, timestep tt; rows >= nrows are padding.
    task check_word;
        input integer which; input integer grp; input integer tt; input integer nrows;
        inout integer errs;
        integer l, row;
        reg [15:0] e, got;
        begin
            for (l = 0; l < 16; l = l + 1) begin
                row = grp * 16 + l;
                if (row < nrows) begin
                    case (which)
                        0: e = xn_exp  [row * T_TOT + tt];
                        1: e = inp_exp [row * T_TOT + tt];
                        2: e = cv_exp  [row * T_TOT + tt];
                        4: e = dt_exp  [row * T_TOT + tt];
                        5: e = da_exp  [row * T_TOT + tt];
                        6: e = yssm_exp[row * T_TOT + tt];
                        7: e = yg_exp  [row * T_TOT + tt];
                        8: e = yn_exp  [row * T_TOT + tt];
                        9: e = op_exp  [row * T_TOT + tt];
                        default: e = xbc_exp[row * T_TOT + tt];
                    endcase
                    got = dma_rdata[l*16 +: 16];
                    if (got !== e || (^got === 1'bx) || (^e === 1'bx)) begin   // x never counts as a match
                        if (errs < 8) $display("  %s MISMATCH t=%0d row=%0d got=%04h exp=%04h",
                                               (which == 0) ? "X_NORM" : (which == 1) ? "INPROJ" : (which == 2) ? "CONV" :
                                               (which == 3) ? "XBC" : (which == 4) ? "DT" : (which == 5) ? "DA" :
                                               (which == 6) ? "Y_SSM" : (which == 7) ? "Y_GATED" : (which == 8) ? "Y_NORM" :
                                               "OUTPROJ", tt, row, got, e);
                        errs = errs + 1;
                    end
                end
            end
        end
    endtask

    task report;
        input [8*16:1] name; input integer n;
        begin
            $display("  %0s errors: %0d / %0d", name, errs, n);
            total_errs = total_errs + errs;
            errs = 0;
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
        $readmemh({`GOLDEN_ROOT, bdir, "P1_Output_Golden_FP.txt"}, p1_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_Norm.txt"},          gam_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_InProj.txt"},        win_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_Norm_Output_FP.txt"},  xn_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_InProj_Out_FP.txt"},   inp_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_Conv.txt"},          wcv_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_B_Conv.txt"},          bcv_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_Conv_Out_FP.txt"},     cv_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_XBC_Silu_FP.txt"},     xbc_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_DtBias.txt"},          dtv_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_A_signed.txt"},        av_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_Delta_FP.txt"},        dt_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_dA_FP.txt"},           da_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_D_param.txt"},         dpar_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_Y_SSM_FP.txt"},        yssm_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_H_State_FP.txt"},      hst_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_GNorm.txt"},         gng_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_Y_Gated_FP.txt"},      yg_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_Y_Norm_FP.txt"},       yn_exp);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_OutProj.txt"},       wo_mem);
        $readmemh({`GOLDEN_ROOT, bdir, "Mam_OutProj_FP.txt"},      op_exp);

        repeat (4) @(negedge clk); rst = 0;
        $display("[M2MIX] block %0d: d_model=%0d d_inner=%0d heads=%0d in_proj rows=%0d (%0d groups) T=%0d",
                 BK, D_MODEL, D_INNER, NH, ROWS, INP_G, T_RUN);

        for (t = 0; t < T_RUN; t = t + 1)
            for (g = 0; g < DM_G; g = g + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = p1_mem[(g*16 + lane)*T_TOT + t];
                dma_wr(3'd0, `PT_INPUT + t*DM_G + g, w);
            end
        for (g = 0; g < DM_G; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = gam_mem[g*16 + lane];
            dma_wr(3'd3, `C_W_NORM_BASE + g, w);
        end
        // in_proj and out_proj: even output groups to cluster 0's weights, odd to cluster 1's, compacted by g/2
        for (g = 0; g < INP_G; g = g + 1)
            for (k = 0; k < D_MODEL; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = ((g*16 + lane) < ROWS) ? win_mem[(g*16 + lane)*D_MODEL + k] : 16'h0000;
                w_put(g % 2, `W_INPROJ_BASE, g / 2, D_MODEL, k, w);
            end
        for (g = 0; g < DM_G; g = g + 1)
            for (k = 0; k < D_INNER; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = wo_mem[(g*16 + lane)*D_INNER + k];
                w_put(g % 2, `W_OUTPROJ_BASE, g / 2, D_INNER, k, w);
            end
        // conv: 4 tap words per global xBC group, in both clusters
        for (g = 0; g < XBC_G; g = g + 1) begin
            for (k = 0; k < 4; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    w[lane*16 +: 16] = wcv_mem[(g*16 + lane)*4 + k];
                dma_wr(3'd2, `W_CONV_BASE + g*4 + k, w);
                dma_wr(3'd4, `W_CONV_BASE + g*4 + k, w);
            end
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = bcv_mem[g*16 + lane];
            dma_wr(3'd3, `C_B_CONV_BASE + g, w);
        end
        // per-head constants, lane = head
        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = (lane < NH) ? dtv_mem[lane] : 16'h0000;
        dma_wr(3'd3, `C_DT_BIAS_BASE, w);
        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = (lane < NH) ? av_mem[lane] : 16'h0000;
        dma_wr(3'd3, `C_A_BASE, w);
        for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = (lane < NH) ? dpar_mem[lane] : 16'h0000;
        dma_wr(3'd3, `C_D_BASE, w);
        for (g = 0; g < DI_G; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1) w[lane*16 +: 16] = gng_mem[g*16 + lane];
            dma_wr(3'd3, `C_W_GNORM_BASE + g, w);
        end
        wx_flush;
        @(negedge clk); dma_write_en = 0;

        @(negedge clk); start = 1; cyc0 = cyc;
        @(negedge clk); start = 0;
        wait (done_all == 1'b1);
        cyc1 = cyc;
        repeat (4) @(negedge clk);
        $display("[M2MIX] compute = %0d cycles (%0d per t)", cyc1 - cyc0, (cyc1 - cyc0) / T_RUN);
        $display("[PROF] per t: RN=%0d M1=%0d M2=%0d M6=%0d GN=%0d M8=%0d barrier=%0d (H_INIT once per block: %0d)",
                 prof[`STG_RN]/T_RUN, prof[`STG_M1A]/T_RUN, prof[`STG_M2]/T_RUN, prof[`STG_M6]/T_RUN,
                 prof[`STG_GN]/T_RUN, prof[`STG_M8]/T_RUN, prof_bar/T_RUN, prof_hinit);

        total_errs = 0; errs = 0;
        for (g = 0; g < DM_G; g = g + 1) begin
            dma_rd((((T_RUN - 1) % 2) ? `PT_X_NORM_B : `PT_X_NORM) + g);   // X_NORM alternates by t parity
            check_word(0, g, T_RUN - 1, D_MODEL, errs);
        end
        report("RN   X_NORM", D_MODEL);
        for (tt = T_RUN - NCHK; tt < T_RUN; tt = tt + 1)
            for (g = 0; g < INP_G; g = g + 1) begin
                dma_rd(`PT_INP_FRAMES + (tt % 4) * INP_G + g);
                check_word(1, g, tt, ROWS, errs);
            end
        report("M1   in_proj", NCHK * ROWS);
        for (g = 0; g < XBC_G; g = g + 1) begin
            dma_rd(`PT_XCONV + g);
            check_word(2, g, T_RUN - 1, XBC_CH, errs);
        end
        report("M2   conv", XBC_CH);
        for (g = 0; g < XBC_G; g = g + 1) begin
            dma_rd(`PT_XBC + g);
            check_word(3, g, T_RUN - 1, XBC_CH, errs);
        end
        report("M2   xBC", XBC_CH);
        dma_rd(`PT_DT);
        check_word(4, 0, T_RUN - 1, NH, errs);
        dma_rd(`PT_DA);
        check_word(5, 0, T_RUN - 1, NH, errs);
        report("M6   dt, dA", 2 * NH);
        for (g = 0; g < DI_G; g = g + 1) begin
            dma_rd(`PT_Y_SSM + g);
            check_word(6, g, T_RUN - 1, D_INNER, errs);
        end
        report("M6   y_ssm", D_INNER);
        if (T_RUN == T_TOT) begin
            // the state of channel hc (group g) is in cluster g%2 at ((g/2)*16 + hc%16) * S_CT + s
            for (hc = 0; hc < D_INNER; hc = hc + 1) begin
                g = hc / 16;
                for (hs = 0; hs < S_CT; hs = hs + 1) begin
                    hword = ((g % 2) == 0) ? dut.u_c0.u_mc.u_hrf.mem[((g / 2) * 16 + (hc % 16)) * S_CT + hs]
                                           : dut.u_c1.u_mc.u_hrf.mem[((g / 2) * 16 + (hc % 16)) * S_CT + hs];
                    for (hl = 0; hl < 16; hl = hl + 1)
                        if (hword[hl*16 +: 16] !== hst_exp[hc * NS + hs * 16 + hl] || (^hword[hl*16 +: 16] === 1'bx)) begin
                            if (errs < 8) $display("  H_STATE MISMATCH ch=%0d n=%0d got=%04h exp=%04h",
                                                   hc, hs * 16 + hl, hword[hl*16 +: 16], hst_exp[hc * NS + hs * 16 + hl]);
                            errs = errs + 1;
                        end
                end
            end
            report("M6   h_state", D_INNER * NS);
        end
        for (g = 0; g < DI_G; g = g + 1) begin
            dma_rd(`PT_Y_GATED + g);
            check_word(7, g, T_RUN - 1, D_INNER, errs);
        end
        report("M6   y_gated", D_INNER);
        for (g = 0; g < DI_G; g = g + 1) begin
            dma_rd(`PT_Y_NORM + g);
            check_word(8, g, T_RUN - 1, D_INNER, errs);
        end
        report("GN   y_norm", D_INNER);
        for (tt = 0; tt < T_RUN; tt = tt + 1)
            for (g = 0; g < DM_G; g = g + 1) begin
                dma_rd(`PT_MAMBA_OUT + tt * DM_G + g);
                check_word(9, g, tt, D_MODEL, errs);
            end
        report("M8   out_proj", D_MODEL * T_RUN);
        dma_read_en = 0;
        mon_report;

        if (total_errs == 0) $display(">>>>> M2MIX BLOCK %0d: BYTE-EXACT PASS <<<<<", BK);
        else                 $display(">>>>> M2MIX BLOCK %0d: FAIL (%0d mismatches) <<<<<", BK, total_errs);
        $finish;
    end

    initial begin
        #800_000_000;
        $display("[M2MIX] TIMEOUT");
        $finish;
    end

`include "m2_write_mon.vh"
endmodule
