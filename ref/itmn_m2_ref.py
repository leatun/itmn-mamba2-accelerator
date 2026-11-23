"""Integer reference of the whole network (raw 12-lead ECG to logits) in numpy, checked against the goldens.

Recomputes every golden file of goldens/itmn_m2 (record 0) from the stored weights and LUT tables alone: encoder,
Inception branches, the Mamba2 mixer (RMSNorm, in_proj, causal conv, SiLU, softplus dt, exp dA, the recurrence,
gate, gated RMSNorm, out_proj), the block output, MaxPool, GAP and FC. All shifts are floors and all saturations
int16, as in the RTL. Exit code 0 if every file matches exactly.

  python ref/itmn_m2_ref.py [golden_root]
"""
import os
import sys

import numpy as np

FB = 11


def sat(x, lo=-32768, hi=32767):
    return np.clip(x, lo, hi)


def load(root, rel, unsigned=False):
    v = np.array([int(l, 16) for l in open(os.path.join(root, rel)) if l.strip()], dtype=np.int64)
    if not unsigned:
        v[v >= 0x8000] -= 0x10000
    return v


def lut(x, table, hi):
    """256-entry floor-indexed LUT over [-8, 8): below -> 0, above -> hi(x)."""
    lo = -8 << FB
    idx = np.clip(((x - lo) * 16) >> FB, 0, 255)
    return np.where(x < lo, 0, np.where(x >= -lo, hi(x), table[idx]))


def rms_v2(x, gamma, rom):
    d = x.shape[0]
    log2d = d.bit_length() - 1
    s = (x * x).sum(0)
    S = rom[np.clip(s >> (log2d + 2 * FB - 1 - 6), 0, 8191)]
    p1 = sat((x * gamma[:, None]) >> FB)
    return sat((p1 * S[None, :]) >> FB)


def rms_fp(x, gamma, rom):
    d = x.shape[0]
    L = d.bit_length() - 1
    s = (x * x).sum(0) + int(round(1e-5 * d * 2 ** (2 * FB)))
    out = np.zeros_like(x)
    p1 = sat((x * gamma[:, None]) >> FB)
    for t in range(x.shape[1]):
        e = int(s[t]).bit_length() - 1
        k = 2 * ((e - 13) // 2)
        m = int(s[t]) >> k if k >= 0 else int(s[t]) << -k
        r = int(rom[(m >> 3) - 1024])
        sh = 10 + k // 2 - L // 2
        v = p1[:, t] * r
        out[:, t] = sat(v >> sh if sh >= 0 else v << -sh)
    return out


def maxpool3(x):
    return np.maximum(np.maximum(np.concatenate([x[:, :1], x[:, :-1]], 1), x), np.concatenate([x[:, 1:], x[:, -1:]], 1))


def conv(x, w, k, pad):
    t = x.shape[1]
    acc = np.zeros((w.shape[0], t), np.int64)
    for tap in range(k):
        sh = tap - pad
        src = np.zeros_like(x)
        if sh >= 0:
            src[:, :t - sh] = x[:, sh:]
        else:
            src[:, -sh:] = x[:, :t + sh]
        acc += w[:, :, tap] @ src
    return sat(acc >> FB)


def main(root):
    bad = []

    def rep(name, got, rel, shape=None, unsigned=False):
        ref = load(root, rel, unsigned).reshape(got.shape)
        ok = np.array_equal(got, ref)
        if not ok:
            bad.append(name)
        print(('OK  ' if ok else 'BAD ') + name + ('' if ok else f'   {int((got != ref).sum())} / {got.size} differ'))

    silu_t, sp_t, exp_t = (load(root, f'luts/{n}_lut.txt') for n in ('silu', 'softplus', 'exp'))
    rom_v2 = load(root, 'luts/rsqrt_v2.txt')
    rom_fp = load(root, 'luts/rsqrt_fp.txt', unsigned=True)

    wav = load(root, 'encoder/Encoder_Input_X.txt').reshape(12, -1)
    enc_w = load(root, 'encoder/Encoder_Weight_Fused.txt').reshape(-1, 12)
    x = sat(sat((enc_w @ wav) >> FB) + load(root, 'encoder/Encoder_Bias_Fused.txt')[:, None])
    rep('encoder', x, 'encoder/Encoder_Output_FP.txt')

    dirs = sorted(d for d in os.listdir(root) if d.startswith('block_'))
    for i, bd in enumerate(dirs):
        g = bd + '/'
        if i in (2, 4):
            x = np.maximum(x[:, 0::2], x[:, 1::2])
        rep(f'B{i} input', x, g + 'P1_Input_X.txt')
        d_in, T = x.shape
        W = lambda n: load(root, g + n + '.txt')
        d_out = W('P1_Bias_Fused').size
        dim = d_out // 4
        p1 = sat(sat((W('P1_Weight_Fused').reshape(d_out, d_in) @ x) >> FB) + W('P1_Bias_Fused')[:, None])
        rep(f'B{i} P1', p1, g + 'P1_Output_Golden_FP.txt')
        bot = sat((W('W_Bot').reshape(dim, d_out) @ p1) >> FB)
        rep(f'B{i} bot', bot, g + 'Out_Bot.txt')
        b1 = sat((W('W_B1').reshape(dim, d_out) @ maxpool3(p1)) >> FB)
        b2, b3, b4 = (conv(bot, W(f'W_B{j}').reshape(dim, dim, k), k, k // 2) for j, k in ((2, 9), (3, 19), (4, 39)))
        for nm, v in (('Out_B1', b1), ('Out_B2', b2), ('Out_B3', b3), ('Out_B4', b4)):
            rep(f'B{i} {nm}', v, g + nm + '.txt')
        inc = np.concatenate([b1, b2, b3, b4], 0)

        # ---- mixer
        d = W('Mam_W_GNorm').size                       # d_inner
        nh = W('Mam_A_signed').size
        n = (W('Mam_B_Conv').size - d) // 2
        p = d // nh
        u = rms_v2(p1, W('Mam_W_Norm'), rom_v2)
        rep(f'B{i} norm', u, g + 'Mam_Norm_Output_FP.txt')
        zx = sat((W('Mam_W_InProj').reshape(d + d + 2 * n + nh, d_out) @ u) >> FB)
        rep(f'B{i} in_proj', zx, g + 'Mam_InProj_Out_FP.txt')
        z, xbc, dtr = zx[:d], zx[d:2 * d + 2 * n], zx[2 * d + 2 * n:]
        wc = W('Mam_W_Conv').reshape(d + 2 * n, 4)
        pad = np.pad(xbc, ((0, 0), (3, 0)))
        acc = sum(pad[:, j:j + T] * wc[:, j:j + 1] for j in range(4))
        cv = sat(sat(acc >> FB) + W('Mam_B_Conv')[:, None])
        rep(f'B{i} conv', cv, g + 'Mam_Conv_Out_FP.txt')
        xbc = lut(cv, silu_t, lambda v: sat(v))
        rep(f'B{i} xBC silu', xbc, g + 'Mam_XBC_Silu_FP.txt')
        xs, Bm, Cm = xbc[:d], xbc[d:d + n], xbc[d + n:]
        dt = lut(sat(dtr + W('Mam_DtBias')[:, None]), sp_t, lambda v: sat(v))
        rep(f'B{i} dt', dt, g + 'Mam_Delta_FP.txt')
        dA = lut(sat((dt * W('Mam_A_signed')[:, None]) >> FB), exp_t, lambda v: np.full_like(v, 32767))
        rep(f'B{i} dA', dA, g + 'Mam_dA_FP.txt')
        D = W('Mam_D_param')
        xh = xs.reshape(nh, p, T)
        h = np.zeros((nh, p, n), np.int64)
        ys = np.zeros((nh, p, T), np.int64)
        for t in range(T):
            xdt = sat((xh[:, :, t] * dt[:, t, None]) >> FB)
            t1 = sat((dA[:, t, None, None] * h) >> FB)
            t2 = sat((xdt[:, :, None] * Bm[None, None, :, t]) >> FB)
            h = sat(t1 + t2)
            ys[:, :, t] = sat(((h * Cm[None, None, :, t]).sum(-1) + D[:, None] * xh[:, :, t]) >> FB)
        y = ys.reshape(d, T)
        rep(f'B{i} y_ssm', y, g + 'Mam_Y_SSM_FP.txt')
        rep(f'B{i} h_last', h, g + 'Mam_H_State_FP.txt')
        yg = sat((y * lut(z, silu_t, lambda v: sat(v))) >> FB)
        rep(f'B{i} y_gated', yg, g + 'Mam_Y_Gated_FP.txt')
        yn = rms_fp(yg, W('Mam_W_GNorm'), rom_fp)
        rep(f'B{i} y_norm', yn, g + 'Mam_Y_Norm_FP.txt')
        mo = sat((W('Mam_W_OutProj').reshape(d_out, d) @ yn) >> FB)
        rep(f'B{i} out_proj', mo, g + 'Mam_OutProj_FP.txt')

        bn = np.maximum(sat(sat((inc * W('Inc_BN_Scale')[:, None]) >> FB) + W('Inc_BN_Shift')[:, None]), 0)
        x = sat(bn + np.maximum(mo, 0))
        rep(f'B{i} final', x, g + 'Final_ITM_Full_FP.txt')

    T = x.shape[1]
    inv = int(round((1 << 15) / T))
    gp = sat((x.sum(1) * inv) >> 15)
    rep('GAP', gp, 'gap_fc/GAP_Output.txt')
    fcw = load(root, 'gap_fc/FC_Weight.txt').reshape(-1, gp.size)
    lg = sat(sat((fcw @ gp) >> FB) + load(root, 'gap_fc/FC_Bias.txt'))
    rep('FC logits', lg, 'gap_fc/FC_Logits.txt')
    print('ALL OK' if not bad else f'MISMATCH in {len(bad)} item(s): {bad}')
    return not bad


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'goldens', 'itmn_m2')) else 1)
