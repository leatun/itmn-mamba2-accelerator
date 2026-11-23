`include "_parameter.v"

// One lane of the PE array, Q4.11, 5 register stages: out_val / out_val2 are valid 5 cycles after the inputs
// (6 after the controller assigns them), acc_raw 4.
//   MAC   acc = (clear ? 0 : acc) + W1*H            out = sat16(acc >> 11)
//   MAC2  acc = (clear ? 0 : acc) + W1*H + W2*X     out = sat16(acc >> 11)
//   MUL   acc = W1*H                                out = sat16(acc >> 11)
//   MUL2  acc = W1*H                                out = sat16(W1*H >> 11), out2 = sat16(W2*X >> 11)
// Stage A registers the inputs, stage M the products (absorbed into the DSP MREG), so the accumulate adder in
// stage B2 is a fabric adder separate from the DSP.
module Mamba_PE_v2 (
    input  wire                      clk,
    input  wire                      rst,
    input  wire [2:0]                op_mode,
    input  wire                      clear_acc,
    input  wire signed [`DATA_W-1:0] in_W1,
    input  wire signed [`DATA_W-1:0] in_H,
    input  wire signed [`DATA_W-1:0] in_W2,
    input  wire signed [`DATA_W-1:0] in_X,
    output reg  signed [`DATA_W-1:0] out_val,
    output reg  signed [`ACC_W-1:0]  acc_raw,
    output reg  signed [`DATA_W-1:0] out_val2
);

    // Stage A
    reg [2:0]                op_mode_a;
    reg                      clear_acc_a;
    reg signed [`DATA_W-1:0] in_W1_a, in_H_a, in_W2_a, in_X_a;

    always @(posedge clk or posedge rst)
        if (rst) begin
            op_mode_a   <= `MAMBA_PE_IDLE;
            clear_acc_a <= 1'b0;
            in_W1_a     <= {`DATA_W{1'b0}};
            in_H_a      <= {`DATA_W{1'b0}};
            in_W2_a     <= {`DATA_W{1'b0}};
            in_X_a      <= {`DATA_W{1'b0}};
        end else begin
            op_mode_a   <= op_mode;
            clear_acc_a <= clear_acc;
            in_W1_a     <= in_W1;
            in_H_a      <= in_H;
            in_W2_a     <= in_W2;
            in_X_a      <= in_X;
        end

    // Stage M
    reg [2:0]                  op_mode_m;
    reg                        clear_acc_m;
    reg signed [2*`DATA_W-1:0] m1_m, m2_m;

    (* use_dsp = "yes" *)
    always @(posedge clk or posedge rst)
        if (rst) begin
            op_mode_m   <= `MAMBA_PE_IDLE;
            clear_acc_m <= 1'b0;
            m1_m        <= {2*`DATA_W{1'b0}};
            m2_m        <= {2*`DATA_W{1'b0}};
        end else begin
            op_mode_m   <= op_mode_a;
            clear_acc_m <= clear_acc_a;
            m1_m        <= in_W1_a * in_H_a;
            m2_m        <= in_W2_a * in_X_a;
        end

    // Stage B1: addend = m1 (+ m2 for MAC2)
    wire signed [2*`DATA_W:0] m12_sum_m = {m1_m[2*`DATA_W-1], m1_m} + {m2_m[2*`DATA_W-1], m2_m};
    wire signed [`ACC_W-1:0]  m1_ext_m  = {{(`ACC_W-2*`DATA_W){m1_m[2*`DATA_W-1]}}, m1_m};
    wire signed [`ACC_W-1:0]  m2_ext_m  = {{(`ACC_W-2*`DATA_W){m2_m[2*`DATA_W-1]}}, m2_m};
    wire signed [`ACC_W-1:0]  m12_ext_m = {{(`ACC_W-(2*`DATA_W+1)){m12_sum_m[2*`DATA_W]}}, m12_sum_m};
    wire signed [`ACC_W-1:0]  addend_m  = (op_mode_m == `MAMBA_PE_MAC2) ? m12_ext_m : m1_ext_m;

    reg [2:0]               op_mode_b1;
    reg                     clear_acc_b1;
    reg signed [`ACC_W-1:0] addend_b1, m2_ext_b1;

    always @(posedge clk or posedge rst)
        if (rst) begin
            op_mode_b1   <= `MAMBA_PE_IDLE;
            clear_acc_b1 <= 1'b0;
            addend_b1    <= {`ACC_W{1'b0}};
            m2_ext_b1    <= {`ACC_W{1'b0}};
        end else begin
            op_mode_b1   <= op_mode_m;
            clear_acc_b1 <= clear_acc_m;
            addend_b1    <= addend_m;
            m2_ext_b1    <= m2_ext_m;
        end

    // Stage B2: accumulate
    wire mac_en_b1  = (op_mode_b1 == `MAMBA_PE_MAC) || (op_mode_b1 == `MAMBA_PE_MAC2);
    wire load_en_b1 = (op_mode_b1 == `MAMBA_PE_MUL) || (op_mode_b1 == `MAMBA_PE_MUL2);
    wire signed [`ACC_W-1:0] acc_next_b1 =
        mac_en_b1  ? (clear_acc_b1 ? {`ACC_W{1'b0}} : acc_raw) + addend_b1 :
        load_en_b1 ? addend_b1 : acc_raw;

    reg [2:0]               op_mode_b;
    reg signed [`ACC_W-1:0] acc_sat_src_b, mul_sat_src_b, mul2_sat_src_b;

    always @(posedge clk or posedge rst)
        if (rst) begin
            acc_raw        <= {`ACC_W{1'b0}};
            op_mode_b      <= `MAMBA_PE_IDLE;
            acc_sat_src_b  <= {`ACC_W{1'b0}};
            mul_sat_src_b  <= {`ACC_W{1'b0}};
            mul2_sat_src_b <= {`ACC_W{1'b0}};
        end else begin
            acc_raw        <= acc_next_b1;
            op_mode_b      <= op_mode_b1;
            acc_sat_src_b  <= acc_next_b1;
            mul_sat_src_b  <= addend_b1;
            mul2_sat_src_b <= m2_ext_b1;
        end

    // Stage C: shift and saturate
    function automatic signed [`DATA_W-1:0] sat16;
        input signed [`ACC_W-1:0] v;
        begin
            sat16 = (v[`ACC_W-1:`DATA_W-1] != {(`ACC_W-`DATA_W+1){v[`ACC_W-1]}})
                  ? (v[`ACC_W-1] ? `SAT16_MIN : `SAT16_MAX) : v[`DATA_W-1:0];
        end
    endfunction

    wire mac_en_b  = (op_mode_b == `MAMBA_PE_MAC) || (op_mode_b == `MAMBA_PE_MAC2);
    wire load_en_b = (op_mode_b == `MAMBA_PE_MUL) || (op_mode_b == `MAMBA_PE_MUL2);

    always @(posedge clk or posedge rst)
        if (rst) begin
            out_val  <= {`DATA_W{1'b0}};
            out_val2 <= {`DATA_W{1'b0}};
        end else begin
            out_val  <= mac_en_b  ? sat16(acc_sat_src_b >>> `FRAC_BITS) :
                        load_en_b ? sat16(mul_sat_src_b >>> `FRAC_BITS) : {`DATA_W{1'b0}};
            out_val2 <= (op_mode_b == `MAMBA_PE_MUL2) ? sat16(mul2_sat_src_b >>> `FRAC_BITS) : {`DATA_W{1'b0}};
        end

endmodule
