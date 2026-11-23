`timescale 1ns / 1ps

// Registered-read RAM. RAM_STYLE "ultra": port A write-only, port B read (URAM cannot be inferred with an
// initial block or a read on the write port). Otherwise: port A write + read-first read, port B read.
module BRAM_256b #(
    parameter ADDR_WIDTH     = 15,
    parameter DATA_WIDTH     = 256,
    parameter RAM_STYLE      = "block",
    parameter DEPTH          = (1 << ADDR_WIDTH),
    parameter CASCADE_HEIGHT = 1024
) (
    input                       clk,
    input                       we_a,
    input      [ADDR_WIDTH-1:0] addr_a,
    input      [DATA_WIDTH-1:0] din_a,
    output reg [DATA_WIDTH-1:0] dout_a,
    input      [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] dout_b
);

    generate
        if (RAM_STYLE == "ultra") begin : URAM_IMPL
            (* ram_style = "ultra", cascade_height = CASCADE_HEIGHT *)
            reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];
            always @(posedge clk) begin
                if (we_a) ram[addr_a] <= din_a;
                dout_a <= {DATA_WIDTH{1'b0}};
                dout_b <= ram[addr_b];
            end
        end else begin : BRAM_IMPL
            (* ram_style = RAM_STYLE, cascade_height = CASCADE_HEIGHT *)
            reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];
            integer i;
            initial
                for (i = 0; i < DEPTH; i = i + 1)
                    ram[i] = {DATA_WIDTH{1'b0}};
            always @(posedge clk) begin
                if (we_a) ram[addr_a] <= din_a;
                dout_a <= ram[addr_a];
                dout_b <= ram[addr_b];
            end
        end
    endgenerate

endmodule
