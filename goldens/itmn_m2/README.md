# Goldens: ITMN with a Mamba2 mixer, Q4.11

Integer weights, input and the expected output of every stage for test record 0 of PTB-XL (fold 10), from the
trained model (d_state 16, headdim 32). `ref/itmn_m2_ref.py` recomputes every file from the weights and tables.

Format: one value per line, 4 hex digits, int16 two's complement (`rsqrt_fp.txt` is unsigned 16-bit); tensors are
row-major (channel, time).

Geometry: blocks 0-3 d_model 64 (d_inner 128, 4 heads, T 1000 / 1000 / 500 / 500); block 4 d_model 128 (d_inner 256,
8 heads, T 250). Conv width 4; MaxPool(2, 2) before blocks 2 and 4. in_proj rows = [z (d_inner) | xBC (d_inner + 32)
| dt (heads)].

| Folder / file | Contents |
|---|---|
| `encoder/` | raw ECG input (12 x 1000), fused 1x1 conv + BN weights and bias, output |
| `block_00_layer00/` .. `block_04_layer06/` | one block each: weights, the block input, every intermediate, the block output |
| `gap_fc/` | GAP output, FC weights and bias, the 5 logits |
| `luts/` | SiLU, softplus and exp tables (256 entries over [-8, 8)), rsqrt ROMs of both RMSNorms (used by the reference) |
| `*_lut.txt`, `rsqrt_*.txt` | the same tables as read by the RTL |

Per block, weights: `P1_Weight_Fused`, `P1_Bias_Fused`, `W_Bot`, `W_B1`..`W_B4`, `Inc_BN_Scale`, `Inc_BN_Shift`,
`Mam_W_Norm` (block RMSNorm gamma), `Mam_W_InProj`, `Mam_W_Conv`, `Mam_B_Conv`, `Mam_DtBias`, `Mam_A_signed` (per
head, negative), `Mam_D_param`, `Mam_W_GNorm` (gated RMSNorm gamma), `Mam_W_OutProj`.

Intermediates: `P1_Input_X`, `P1_Output_Golden_FP`, `Out_Bot`, `Out_B1`..`Out_B4`, `Mam_Norm_Output_FP`,
`Mam_InProj_Out_FP`, `Mam_Conv_Out_FP` (conv + bias, before SiLU), `Mam_XBC_Silu_FP`, `Mam_Delta_FP` (heads x T),
`Mam_dA_FP` (heads x T), `Mam_Y_SSM_FP`, `Mam_Y_Gated_FP`, `Mam_Y_Norm_FP`, `Mam_OutProj_FP`, `Mam_H_State_FP`
(state after the last timestep), `Final_ITM_Full_FP` (block output).

Arithmetic: all shifts are floors and all saturations int16. The exact formulas, including both RMSNorms, are in
`ref/itmn_m2_ref.py`.
