#!/usr/bin/env python3
"""Split-K model of the MAC jobs: the PE array and the scan rows (aux engine) share every output group.

A job computes y[g] = W[g] . x for 16 output rows per group over w = len/16 input words. The PE takes words
[0, wp) with wp = ceil(w/2), lane = output row, weights in bank P in the normal layout (word = base + g*lenP + k,
lane r = W[16g+r][k]). The aux engine takes words [wp, w), lane = input element, weights in bank X transposed
(word = base + g*lenX + (j-wp)*16 + r, lane i = W[16g+r][16j+i]); each of its two adder trees reduces one row per
cycle (rows 0-7 and 8-15). Output = sat16((acc_PE + acc_aux) >> 11), then the bias as today.

The script builds both banks from the golden weights through those address formulas, runs every MAC job of
blocks 0-4 (and the encoder and FC) through the two engines word by word, and compares with the goldens. It also
derives the bank bases and checks that every phase fits in a 4,096-word bank.

Usage:  python splitk_model.py [golden_root]
"""
import os
import sys
import numpy as np

FB = 11
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'goldens', 'itmn_m2')
BANK_DEPTH = 4096
W_FC_BASE = 3584                                     # _parameter.v
DIMS = {0: (64, 64), 1: (64, 64), 2: (64, 64), 3: (64, 64), 4: (64, 128)}   # (d_in, d_model) per block
POOL_BEFORE = {2, 4}


def sat(x):
    return np.clip(x, -32768, 32767)


def load(rel):
    v = np.array([int(l, 16) for l in open(os.path.join(ROOT, rel)) if l.strip()], dtype=np.int64)
    v[v >= 0x8000] -= 0x10000
    return v


def split(w):
    wp = (w + 1) // 2
    return wp, w - wp


class Bank:
    def __init__(self, name):
        self.name, self.mem = name, {}

    def wr(self, addr, word):
        assert 0 <= addr < BANK_DEPTH, f'{self.name}: address {addr} outside the bank'
        assert addr not in self.mem, f'{self.name}: address {addr} written twice'
        self.mem[addr] = np.asarray(word, np.int64)

    def rd(self, addr):
        return self.mem[addr]


def load_job(bp, bx, base_p, base_x, Wg):
    """Write group matrices Wg[g] (16 x len) into banks P and X; return (lenP, lenX)."""
    ln = Wg.shape[2]
    w = ln // 16
    wp, wx = split(w)
    lp, lx = wp * 16, wx * 16
    for g in range(Wg.shape[0]):
        for k in range(lp):
            bp.wr(base_p + g * lp + k, Wg[g, :, k])
        for jj in range(wx):
            j = wp + jj
            for r in range(16):
                bx.wr(base_x + g * lx + jj * 16 + r, Wg[g, r, 16 * j:16 * j + 16])
    return lp, lx


def run_job(bp, bx, base_p, base_x, n_grp, ln, xwords):
    """xwords: (n_t, w, 16) input words per output timestep (padding already zero). Returns raw sums (n_t, n_grp*16)."""
    w = ln // 16
    wp, wx = split(w)
    lp, lx = wp * 16, wx * 16
    n_t = xwords.shape[0]
    out = np.zeros((n_t, n_grp * 16), np.int64)
    for g in range(n_grp):
        # PE: lane r accumulates W[r][k] * x[k], two elements per cycle
        WP = np.stack([bp.rd(base_p + g * lp + k) for k in range(lp)], 1)          # 16 x lp
        acc_pe = xwords[:, :wp, :].reshape(n_t, lp) @ WP.T                         # n_t x 16
        # aux: per word, tree A rows 0-7 and tree B rows 8-15, one row per cycle
        acc_aux = np.zeros((n_t, 16), np.int64)
        for jj in range(wx):
            xw = xwords[:, wp + jj, :]                                             # n_t x 16 (lane = input)
            for c in range(8):
                for r in (c, c + 8):
                    acc_aux[:, r] += xw @ bx.rd(base_x + g * lx + jj * 16 + r)
        out[:, g * 16:(g + 1) * 16] = acc_pe + acc_aux
    return out


def words_of(x):
    """(C, T) channel-major -> (T, C/16, 16) words per t."""
    c, t = x.shape
    return x.T.reshape(t, c // 16, 16)


def conv_words(xw, k, pad):
    """Input words of a K-tap conv per output t: element tap*dim + ci = x[ci][t - pad + tap], zero outside [0, T)."""
    t, nw, _ = xw.shape
    z = np.zeros((pad, nw, 16), np.int64)
    zp = np.concatenate([z, xw, np.zeros((k - 1 - pad, nw, 16), np.int64)], 0)
    return np.stack([zp[i:i + k].reshape(k * nw, 16) for i in range(t)], 0)


def group_mats(W):
    rows, ln = W.shape
    assert rows % 16 == 0 and ln % 16 == 0
    return W.reshape(rows // 16, 16, ln)


def maxpool3(x):
    left = np.concatenate([x[:, :1], x[:, :-1]], 1)
    right = np.concatenate([x[:, 1:], x[:, -1:]], 1)
    return np.maximum(np.maximum(left, x), right)


def check(label, got, want):
    ok = np.array_equal(got, want)
    print(f'  {"OK " if ok else "BAD"} {label}' + ('' if ok else f'  ({np.count_nonzero(got != want)} diffs)'))
    return ok


# ---------------------------------------------------------------------------------------------------------------
# Bank bases. Inception phase (weights shared by both clusters) and Mamba phase (per-cluster, groups compacted)
# are separate loads. Each region is sized for the largest block; the same base is used in both banks.
# ---------------------------------------------------------------------------------------------------------------
def region(n_grp, w):
    wp, wx = split(w)
    return n_grp * wp * 16, n_grp * wx * 16


def bases():
    inc = {}
    for i, (d_in, d_model) in DIMS.items():
        dim = d_model // 4
        cw = d_model // 16
        jobs = {'P1': (cw, d_in // 16), 'BOT': (cw // 4, cw), 'B1': (cw // 4, cw),
                'B2': (cw // 4, 9 * dim // 16), 'B3': (cw // 4, 19 * dim // 16), 'B4': (cw // 4, 39 * dim // 16)}
        if i == 0:
            jobs['ENC'] = (d_in // 16, 1)
        for n, (g, w) in jobs.items():
            p, x = region(g, w)
            inc[n] = (max(inc.get(n, (0, 0))[0], p), max(inc.get(n, (0, 0))[1], x))
    out = {}
    cur = 0
    for n in ('P1', 'BOT', 'B1', 'B2', 'B3', 'B4', 'ENC'):
        out['W_' + n] = cur
        cur += max(inc[n])
    inc_end = cur
    return out, inc, inc_end


def mamba_bases(S_list=(1, 2, 4, 8)):
    """Mamba phase per cluster: conv taps (bank P only), in_proj / out_proj halves, FC (cluster 0)."""
    res = {}
    for S in S_list:
        need = {}
        for i, (_, d_model) in DIMS.items():
            d_inner = 2 * d_model
            xbc_g = (d_inner + 32 * S) // 16
            inp_g = d_inner // 16 + xbc_g + 1
            out_g = d_model // 16
            for n, (g, w) in {'INPROJ': ((inp_g + 1) // 2, d_model // 16),
                              'OUTPROJ': ((out_g + 1) // 2, d_inner // 16),
                              'FC': (1, 8)}.items():
                p, x = region(g, w)
                need[n] = (max(need.get(n, (0, 0))[0], p), max(need.get(n, (0, 0))[1], x))
            need['CONV'] = (max(need.get('CONV', (0, 0))[0], xbc_g * 4), 0)
        cur, b = 0, {}
        for n in ('CONV', 'INPROJ', 'OUTPROJ'):
            b['W_' + n] = cur
            cur += (max(need[n]) + 15) // 16 * 16             # regions start on multiples of 16, as in m2_layout.py
        assert cur <= W_FC_BASE, f'S={S}: Mamba phase reaches W_FC_BASE'
        b['W_FC'] = W_FC_BASE
        cur = W_FC_BASE + max(need['FC'])
        assert cur <= BANK_DEPTH, f'S={S}: Mamba phase needs {cur} words per bank'
        res[S] = (b, cur)
    return res


def main():
    ok = True
    inc_b, inc_need, inc_end = bases()
    assert inc_end <= BANK_DEPTH, f'Inception phase needs {inc_end} words per bank'
    print(f'Inception phase bank bases {inc_b}, end {inc_end} of {BANK_DEPTH}')
    for S, (b, end) in mamba_bases().items():
        print(f'Mamba phase S={S}: bases {b}, end {end} of {BANK_DEPTH}')
    mb = mamba_bases((1,))[1][0]

    blocks = sorted(d for d in os.listdir(ROOT) if d.startswith('block_'))
    for i, bd in enumerate(blocks):
        d_in, d_model = DIMS[i]
        dim = d_model // 4
        g = bd + '/'
        x_in = load(g + 'P1_Input_X.txt').reshape(d_in, -1)
        T = x_in.shape[1]
        print(f'block {i} ({bd}, T={T})')
        # Inception phase: one load of both banks
        bp, bx = Bank('P'), Bank('X')
        WP1 = group_mats(load(g + 'P1_Weight_Fused.txt').reshape(d_model, d_in))
        Wbot = group_mats(load(g + 'W_Bot.txt').reshape(dim, d_model))
        Wb1 = group_mats(load(g + 'W_B1.txt').reshape(dim, d_model))
        Wk = {}
        for n, k in (('B2', 9), ('B3', 19), ('B4', 39)):
            w3 = load(g + f'W_{n}.txt').reshape(dim, dim, k)
            Wk[n] = (k, group_mats(w3.transpose(0, 2, 1).reshape(dim, k * dim)))   # element tap*dim + ci
        for n, Wg in (('P1', WP1), ('BOT', Wbot), ('B1', Wb1)) + tuple((n, v[1]) for n, v in Wk.items()):
            load_job(bp, bx, inc_b['W_' + n], inc_b['W_' + n], Wg)

        p1_raw = run_job(bp, bx, inc_b['W_P1'], inc_b['W_P1'], WP1.shape[0], d_in, words_of(x_in))
        p1 = sat(sat(p1_raw >> FB) + load(g + 'P1_Bias_Fused.txt')[None, :]).T
        ok &= check('P1', p1, load(g + 'P1_Output_Golden_FP.txt').reshape(d_model, T))
        bot = sat(run_job(bp, bx, inc_b['W_BOT'], inc_b['W_BOT'], Wbot.shape[0], d_model, words_of(p1)) >> FB).T
        ok &= check('BOT', bot, load(g + 'Out_Bot.txt').reshape(dim, T))
        b1 = sat(run_job(bp, bx, inc_b['W_B1'], inc_b['W_B1'], Wb1.shape[0], d_model,
                         words_of(maxpool3(p1))) >> FB).T
        ok &= check('B1', b1, load(g + 'Out_B1.txt').reshape(dim, T))
        for n, (k, Wg) in Wk.items():
            xw = conv_words(words_of(bot), k, k // 2)
            y = sat(run_job(bp, bx, inc_b['W_' + n], inc_b['W_' + n], Wg.shape[0], k * dim, xw) >> FB).T
            ok &= check(f'{n} (k{k}, w={k * dim // 16})', y, load(g + f'Out_{n}.txt').reshape(dim, T))

        # Mamba phase, cluster c holds output groups g with g % 2 == c (compacted)
        d_inner = 2 * d_model
        Win = load(g + 'Mam_W_InProj.txt').reshape(-1, d_model)
        Wout = load(g + 'Mam_W_OutProj.txt').reshape(d_model, d_inner)
        rows = Win.shape[0]
        Win = np.concatenate([Win, np.zeros(((-rows) % 16, d_model), np.int64)], 0)
        xn = load(g + 'Mam_Norm_Output_FP.txt').reshape(d_model, T)
        yn = load(g + 'Mam_Y_Norm_FP.txt').reshape(d_inner, T)
        inp = np.zeros((T, Win.shape[0]), np.int64)
        outp = np.zeros((T, d_model), np.int64)
        for c in (0, 1):
            bp, bx = Bank('P'), Bank('X')
            Gi = group_mats(Win)[c::2]
            Go = group_mats(Wout)[c::2]
            load_job(bp, bx, mb['W_INPROJ'], mb['W_INPROJ'], Gi)
            load_job(bp, bx, mb['W_OUTPROJ'], mb['W_OUTPROJ'], Go)
            ri = sat(run_job(bp, bx, mb['W_INPROJ'], mb['W_INPROJ'], Gi.shape[0], d_model, words_of(xn)) >> FB)
            ro = sat(run_job(bp, bx, mb['W_OUTPROJ'], mb['W_OUTPROJ'], Go.shape[0], d_inner, words_of(yn)) >> FB)
            for gi in range(Gi.shape[0]):
                inp[:, (2 * gi + c) * 16:(2 * gi + c + 1) * 16] = ri[:, gi * 16:(gi + 1) * 16]
            for gi in range(Go.shape[0]):
                outp[:, (2 * gi + c) * 16:(2 * gi + c + 1) * 16] = ro[:, gi * 16:(gi + 1) * 16]
        ok &= check(f'in_proj ({rows} rows, w={d_model // 16})', inp[:, :rows].T,
                    load(g + 'Mam_InProj_Out_FP.txt').reshape(rows, T))
        ok &= check(f'out_proj (w={d_inner // 16})', outp.T, load(g + 'Mam_OutProj_FP.txt').reshape(d_model, T))

    # Encoder (w = 1: PE only) and FC (w = 8)
    print('head')
    bp, bx = Bank('P'), Bank('X')
    wav = load('encoder/Encoder_Input_X.txt').reshape(12, -1)
    wav16 = np.concatenate([wav, np.zeros((4, wav.shape[1]), np.int64)], 0)
    Wenc = load('encoder/Encoder_Weight_Fused.txt').reshape(64, 12)
    Wenc = np.concatenate([Wenc, np.zeros((64, 4), np.int64)], 1)
    load_job(bp, bx, inc_b['W_ENC'], inc_b['W_ENC'], group_mats(Wenc))
    enc = sat(sat(run_job(bp, bx, inc_b['W_ENC'], inc_b['W_ENC'], 4, 16, words_of(wav16)) >> FB)
              + load('encoder/Encoder_Bias_Fused.txt')[None, :]).T
    ok &= check('encoder (w=1, PE only)', enc, load('encoder/Encoder_Output_FP.txt').reshape(64, -1))
    bp, bx = Bank('P'), Bank('X')
    Wfc = load('gap_fc/FC_Weight.txt').reshape(-1, 128)
    nc = Wfc.shape[0]
    Wfc = np.concatenate([Wfc, np.zeros((16 - nc, 128), np.int64)], 0)
    load_job(bp, bx, mb['W_FC'], mb['W_FC'], group_mats(Wfc))
    gapv = load('gap_fc/GAP_Output.txt')
    lg = sat(sat(run_job(bp, bx, mb['W_FC'], mb['W_FC'], 1, 128, words_of(gapv[:, None])) >> FB)[0, :nc]
             + load('gap_fc/FC_Bias.txt'))
    ok &= check('FC (w=8)', lg, load('gap_fc/FC_Logits.txt'))

    print('SPLIT-K MODEL: ALL BYTE-EXACT' if ok else 'SPLIT-K MODEL: MISMATCH')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
