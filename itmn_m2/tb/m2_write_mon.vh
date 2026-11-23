// Write monitor of tb_M2_MIX. Every main-RAM write of both clusters is
// checked against the goldens at the timestep it belongs to (the end-of-run checks see most stages at the last t
// only), X_NORM buffer parity and in_proj frame index included; the scan's gate row must have at most one requester
// per cycle and a matching registered operand select. mon_report also checks the number of writes of each kind and
// that cluster 1's mirrored copy equals cluster 0's RAM, and adds its error count to total_errs.
integer a_n [0:9];
integer a_e [0:9];
integer a_coll, a_sel, a_sqcoll, a_unk;
initial begin : a_init
    integer i;
    for (i = 0; i < 10; i = i + 1) begin a_n[i] = 0; a_e[i] = 0; end
    a_coll = 0; a_sel = 0; a_sqcoll = 0; a_unk = 0;
end

task a_cmp;
    input integer which; input integer grp; input integer tt; input integer nrows; input [255:0] data;
    input integer cl;
    integer l, row, bad;
    reg [15:0] e, got;
    begin
        bad = 0;
        for (l = 0; l < 16; l = l + 1) begin
            row = grp * 16 + l;
            if (row < nrows && tt >= 0 && tt < T_TOT) begin
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
                got = data[l*16 +: 16];
                if (got !== e || (^got === 1'bx) || (^e === 1'bx)) bad = bad + 1;
            end
        end
        a_n[which] = a_n[which] + 1;
        if (tt < 0 || tt >= T_RUN) a_unk = a_unk + 1;
        if (bad) begin
            if (a_e[which] < 4) $display("  [MON] write mismatch kind=%0d cl=%0d grp=%0d t=%0d (%0d lanes)", which, cl, grp, tt, bad);
            a_e[which] = a_e[which] + 1;
        end
    end
endtask

task a_write;
    input integer cl; input [14:0] a; input [255:0] d; input integer tc; input hoist;
    integer tt, f;
    begin
        if (a >= `PT_X_NORM && a < `PT_X_NORM + DM_G) begin
            tt = hoist ? tc + 1 : tc;
            if (tt % 2) a_unk = a_unk + 1;          // odd t must go to buffer B
            a_cmp(0, a - `PT_X_NORM, tt, D_MODEL, d, cl);
        end else if (a >= `PT_X_NORM_B && a < `PT_X_NORM_B + DM_G) begin
            tt = hoist ? tc + 1 : tc;
            if (!(tt % 2)) a_unk = a_unk + 1;
            a_cmp(0, a - `PT_X_NORM_B, tt, D_MODEL, d, cl);
        end else if (a >= `PT_INP_FRAMES && a < `PT_INP_FRAMES + 4 * INP_G) begin
            f = (a - `PT_INP_FRAMES) / INP_G;
            if (f != tc % 4) a_unk = a_unk + 1;
            a_cmp(1, (a - `PT_INP_FRAMES) % INP_G, tc, ROWS, d, cl);
        end else if (a >= `PT_XCONV && a < `PT_XCONV + XBC_G) a_cmp(2, a - `PT_XCONV, tc, XBC_CH, d, cl);
        else if (a >= `PT_XBC && a < `PT_XBC + XBC_G)       a_cmp(3, a - `PT_XBC, tc, XBC_CH, d, cl);
        else if (a == `PT_DT)                               a_cmp(4, 0, tc, NH, d, cl);
        else if (a == `PT_DA)                               a_cmp(5, 0, tc, NH, d, cl);
        else if (a >= `PT_Y_SSM && a < `PT_Y_SSM + DI_G)     a_cmp(6, a - `PT_Y_SSM, tc, D_INNER, d, cl);
        else if (a >= `PT_Y_GATED && a < `PT_Y_GATED + DI_G) a_cmp(7, a - `PT_Y_GATED, tc, D_INNER, d, cl);
        else if (a >= `PT_Y_NORM && a < `PT_Y_NORM + DI_G)   a_cmp(8, a - `PT_Y_NORM, tc, D_INNER, d, cl);
        else if (a >= `PT_MAMBA_OUT && a < `PT_MAMBA_OUT + T_TOT * DM_G)
            a_cmp(9, (a - `PT_MAMBA_OUT) % DM_G, (a - `PT_MAMBA_OUT) / DM_G, D_MODEL, d, cl);
        else a_unk = a_unk + 1;
    end
endtask

// m_we is registered: the write seen now was decided in the previous cycle, so use that cycle's t and hoist flag
integer a_tc0, a_tc1; reg a_h0, a_h1;
initial begin a_tc0 = 0; a_tc1 = 0; a_h0 = 0; a_h1 = 0; end
always @(posedge clk) begin
    if (dut.u_c0.m_we) a_write(0, dut.u_c0.m_wr_addr, dut.u_c0.m_wr_data, a_tc0, a_h0);
    if (dut.u_c1.m_we) a_write(1, dut.u_c1.m_wr_addr, dut.u_c1.m_wr_data, a_tc1, a_h1);
    a_tc0 = dut.u_c0.t_cnt; a_h0 = dut.u_c0.rn_hoist;
    a_tc1 = dut.u_c1.t_cnt; a_h1 = dut.u_c1.rn_hoist;
end

// Gate row: at most one requester per cycle, and the registered select must match it
task a_gate;
    input ax, gg, gx, sg, sr, gd, rl; input [1:0] sel; input integer cl;
    begin
        if (ax + gg + gx + sg + sr > 1) begin
            if (a_coll < 4) $display("  [MON] gate-row collision cl=%0d ax=%0d g=%0d x=%0d sqg=%0d sqr=%0d", cl, ax, gg, gx, sg, sr);
            a_coll = a_coll + 1;
        end
        if ((gg && sel != 2'd1) || (gx && !ax && sel != 2'd3) || ((sg || sr) && sel != 2'd2) || (ax && sel != 2'd0)) begin
            if (a_sel < 4) $display("  [MON] gate-row select mismatch cl=%0d sel=%0d", cl, sel);
            a_sel = a_sel + 1;
        end
        if (gd && rl) a_sqcoll = a_sqcoll + 1;
    end
endtask
always @(posedge clk) begin
    a_gate(dut.u_c0.ax_on_r, dut.u_c0.gt_go_g, dut.u_c0.gt_go_x, dut.u_c0.sq_g, dut.u_c0.sq_r,
           dut.u_c0.gt_go_gd, dut.u_c0.rn_ld_r, dut.u_c0.gt_sel, 0);
    a_gate(dut.u_c1.ax_on_r, dut.u_c1.gt_go_g, dut.u_c1.gt_go_x, dut.u_c1.sq_g, dut.u_c1.sq_r,
           dut.u_c1.gt_go_gd, dut.u_c1.rn_ld_r, dut.u_c1.gt_sel, 1);
end

// Writes expected per run, per kind (dt and dA are written by both clusters)
function integer mon_expect;
    input integer which;
    case (which)
        0, 9:       mon_expect = T_RUN * DM_G;
        1:          mon_expect = T_RUN * INP_G;
        2, 3:       mon_expect = T_RUN * XBC_G;
        4, 5:       mon_expect = T_RUN * 2;
        default:    mon_expect = T_RUN * DI_G;
    endcase
endfunction

// Both clusters' main RAMs must hold the same mirrored area at the end ([PT_MIX_END, PT_INPUT) is not mirrored)
task mon_mirror_check;
    integer a, bad;
    begin
        bad = 0;
        for (a = 0; a < `PT_MAIN_DEPTH; a = a + 1)
            if ((a < `PT_MIX_END || a >= `PT_INPUT) &&
                dut.u_mem0.u_ram_main.URAM_IMPL.ram[a] !== dut.u_mem1.u_ram_main.URAM_IMPL.ram[a]) begin
                if (bad < 4) $display("  [MON] mirror copies differ at %0d", a);
                bad = bad + 1;
            end
        $display("[MON] mirror copies: %0d words differ", bad);
        total_errs = total_errs + bad;
    end
endtask

task mon_report;
    integer i, tot, cnt;
    begin
        tot = 0; cnt = 0;
        for (i = 0; i < 10; i = i + 1) begin
            tot = tot + a_e[i];
            if (a_n[i] != mon_expect(i)) begin
                $display("  [MON] kind %0d: %0d writes, expected %0d", i, a_n[i], mon_expect(i));
                cnt = cnt + 1;
            end
        end
        mon_mirror_check;
        total_errs = total_errs + tot + cnt + a_unk + a_coll + a_sel + a_sqcoll;
        $display("[MON] writes checked: xn=%0d inp=%0d conv=%0d xbc=%0d dt=%0d dA=%0d yssm=%0d yg=%0d yn=%0d op=%0d",
                 a_n[0], a_n[1], a_n[2], a_n[3], a_n[4], a_n[5], a_n[6], a_n[7], a_n[8], a_n[9]);
        $display("[MON] write mismatches: %0d  unknown/misplaced writes: %0d  gate collisions: %0d  select errors: %0d  sq_op collisions: %0d",
                 tot, a_unk, a_coll, a_sel, a_sqcoll);
    end
endtask
