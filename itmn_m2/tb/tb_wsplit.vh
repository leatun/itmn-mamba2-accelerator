// MAC weight loading in the split-K layout of Mamba_Cluster (included in the module body after task dma_wr).
// Job (base, output group g in this RAM, len input elements): with w = len/16 words the PE gets words [0, wp),
// wp = ceil(w/2), in bank P at base + g*lenp + k (lane = output row), and the aux engine words [wp, w) in bank X
// (weight address + W_MEM_DEPTH) transposed: word base + g*lenx + (j - wp)*16 + r holds W[16g + r][16j .. 16j + 15].
// Bank X words are assembled in wx_sh and written by wx_flush.
// cl: 0 or 1 for one cluster's RAM, 2 for both.

reg [255:0] wx_sh   [0:1][0:`W_MEM_DEPTH-1];
reg         wx_used [0:1][0:`W_MEM_DEPTH-1];

initial begin : wx_init
    integer c_, a_;
    for (c_ = 0; c_ < 2; c_ = c_ + 1)
        for (a_ = 0; a_ < `W_MEM_DEPTH; a_ = a_ + 1)
            wx_used[c_][a_] = 1'b0;
end

task w_put;
    input integer cl; input integer base; input integer g; input integer len; input integer k;
    input [255:0] data;
    integer w_, wp_, lp_, lx_, c_, r_, a_;
    begin
        w_  = len / 16;
        wp_ = (w_ + 1) / 2;
        lp_ = wp_ * 16;
        lx_ = len - lp_;
        if (k < lp_) begin
            if (cl != 1) dma_wr(3'd2, base + g*lp_ + k, data);
            if (cl != 0) dma_wr(3'd4, base + g*lp_ + k, data);
        end else
            for (c_ = 0; c_ < 2; c_ = c_ + 1)
                if (cl == 2 || cl == c_)
                    for (r_ = 0; r_ < 16; r_ = r_ + 1) begin
                        a_ = base + g*lx_ + (k/16 - wp_)*16 + r_;
                        wx_sh[c_][a_][(k%16)*16 +: 16] = data[r_*16 +: 16];
                        wx_used[c_][a_] = 1'b1;
                    end
    end
endtask

task wx_flush;
    integer c_, a_;
    begin
        for (c_ = 0; c_ < 2; c_ = c_ + 1)
            for (a_ = 0; a_ < `W_MEM_DEPTH; a_ = a_ + 1)
                if (wx_used[c_][a_]) begin
                    dma_wr(c_ == 0 ? 3'd2 : 3'd4, `W_MEM_DEPTH + a_, wx_sh[c_][a_]);
                    wx_used[c_][a_] = 1'b0;
                end
    end
endtask
