#!/usr/bin/env python3
"""Memory layout of the Mamba2 mixer for d_state = 16*S (S = 1, 2, 4, 8), with overlap and capacity checks.

Source of the constants in src/common/m2_layout.vh. Run it after changing the geometry; it asserts that nothing
overlaps and that every block (0..4) fits, for every S.
  * 16 lanes per 256-bit word; a group = 16 channels = 1 word.
  * The mixer area of ram_main is indexed by global group and mirrored to both clusters.
  * Output groups of in_proj, out_proj and the scan alternate between the clusters (even -> 0, odd -> 1).
  * B and C take S words each in the xBC frame (x | B | C = d_inner/16 + 2*S groups).

Usage:  python m2_layout.py            check, and compare the S = 1 branch of m2_layout.vh with the script
        python m2_layout.py --write    regenerate m2_layout.vh
"""
import os
import re
import sys

HEADDIM_MIN, D_CONV = 16, 4
W_BANK = 4096                                        # words per weight bank (P and X)
LAYOUT_VH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'common', 'm2_layout.vh')
G = lambda n: (n + 15) // 16


def geom(d_model, S):
    d_inner = 2 * d_model
    xbc_g = G(d_inner + 2 * 16 * S)                  # x | B | C
    nh_max = d_inner // HEADDIM_MIN                  # heads at the smallest head size
    assert nh_max <= 16, 'dt / dA words hold one head per lane'
    inp_g = G(d_inner) + xbc_g + 1                   # z | xBC | dt
    return dict(d_model=d_model, d_in_g=G(d_model), d_inner=d_inner, xbc_g=xbc_g, dt_g=1, inp_g=inp_g,
                y_g=G(d_inner), out_g=G(d_model))


def overlaps(table, limit, label):
    items = sorted(table.items(), key=lambda kv: kv[1][0])
    for (a, (sa, za)), (b, (sb, zb)) in zip(items, items[1:]):
        assert sa + za <= sb, f'{label}: {a} [{sa},{sa + za}) overlaps {b} [{sb},{sb + zb})'
    assert items[-1][1][0] + items[-1][1][1] <= limit, f'{label}: past {limit}'


def layout(S):
    blocks = [geom(64, S)] * 4 + [geom(128, S)]
    maxg = {k: max(b[k] for b in blocks) for k in blocks[0]}
    R = dict(S=S, blocks=blocks, maxg=maxg)

    # --- ram_main mixer area
    main, cur = {}, 0

    def alloc(name, size):
        nonlocal cur
        main[name] = (cur, size)
        cur += size

    FRAME = maxg['inp_g']
    alloc('X_NORM', maxg['d_in_g'])
    alloc('INP_FRAMES', 4 * FRAME)
    alloc('XBC', maxg['xbc_g'])
    alloc('DT', 1)
    alloc('DA', 1)
    alloc('Y_SSM', maxg['y_g'])
    alloc('Y_GATED', maxg['y_g'])
    alloc('Y_NORM', maxg['y_g'])
    alloc('XCONV', maxg['xbc_g'])
    alloc('X_NORM_B', maxg['d_in_g'])                 # X_NORM of odd t
    MIX_END = G(cur) * 16
    BULK = 256 if S == 1 else max(256, MIX_END)
    assert MIX_END <= BULK
    MAIN_DEPTH = BULK + 12000
    assert MAIN_DEPTH <= (12288 if S == 1 else 16384), f'ram_main depth {MAIN_DEPTH}'
    R.update(main=main, FRAME=FRAME, MIX_END=MIX_END, BULK=BULK, MAIN_DEPTH=MAIN_DEPTH)

    # --- weight RAM (Mamba phase), per bank: a MAC job of w input words per group needs 16*ceil(w/2) words per
    # group in bank P and 16*floor(w/2) in bank X (split-K); regions start on multiples of 16
    W, wcur = {}, 0

    def walloc(name, size):
        nonlocal wcur
        W[name] = (wcur, size)
        wcur += G(size) * 16

    half = lambda n: (n + 1) // 2
    walloc('W_CONV', maxg['xbc_g'] * D_CONV)
    walloc('W_INPROJ', max(half(b['inp_g']) * 16 * half(b['d_model'] // 16) for b in blocks))
    walloc('W_OUTPROJ', max(half(b['out_g']) * 16 * half(b['d_inner'] // 16) for b in blocks))
    W_FC = 3584
    assert wcur <= W_FC, f'S={S}: Mamba-phase weights {wcur} collide with W_FC {W_FC}'
    R.update(W=W, W_FC=W_FC, wcur=wcur)

    # --- const RAM
    C, ccur = {}, 0

    def calloc(name, size):
        nonlocal ccur
        C[name] = (ccur, size)
        ccur += size

    calloc('C_W_NORM', maxg['d_in_g'])
    calloc('C_W_GNORM', maxg['y_g'])
    calloc('C_B_CONV', maxg['xbc_g'])
    calloc('C_DT_BIAS', 1)
    calloc('C_A', 1)
    calloc('C_D', 1)
    assert ccur <= 64, f'S={S}: Mamba consts {ccur} reach the INC consts at 64'
    R.update(C=C)

    # --- H register file: per cluster d_inner/2 channels x S words (block 4)
    h_need = (maxg['d_inner'] // 2) * S
    R.update(H_DEPTH=max(256, 1 << (h_need - 1).bit_length()))
    R['H_ADDR_W'] = max(9, (R['H_DEPTH'] - 1).bit_length())

    overlaps(main, BULK, f'S={S} ram_main mixer area')
    overlaps(W, W_BANK, f'S={S} weight bank')
    overlaps(C, 64, f'S={S} const RAM')
    return R


def defines(R):
    m, W, C = R['main'], R['W'], R['C']
    d = [('H_ADDR_W', R['H_ADDR_W']), ('H_DEPTH', R['H_DEPTH']),
         ('PT_X_NORM', m['X_NORM'][0]), ('PT_INP_FRAMES', m['INP_FRAMES'][0]), ('PT_INP_FRAME_MAX', R['FRAME']),
         ('PT_XBC', m['XBC'][0]), ('PT_DT', m['DT'][0]), ('PT_DA', m['DA'][0]), ('PT_Y_SSM', m['Y_SSM'][0]),
         ('PT_Y_GATED', m['Y_GATED'][0]), ('PT_Y_NORM', m['Y_NORM'][0]), ('PT_XCONV', m['XCONV'][0]), ('PT_X_NORM_B', m['X_NORM_B'][0]),
         ('PT_MIX_END', R['MIX_END']),
         ('PT_INPUT', R['BULK']), ('PT_MAMBA_OUT', R['BULK']), ('PT_X_IN', R['BULK'] + 4000), ('PT_INC_OUT', R['BULK'] + 4000),
         ('PT_MP3', R['BULK'] + 8000), ('PT_BOT', R['BULK'] + 8000),
         ('PT_ENC_IN', R['BULK'] + 8000), ('PT_GAP', R['BULK'] + 8000), ('PT_LOGITS', R['BULK'] + 8008), ('PT_MAIN_DEPTH', R['MAIN_DEPTH']),
         ('W_CONV_BASE', W['W_CONV'][0]), ('W_INPROJ_BASE', W['W_INPROJ'][0]), ('W_OUTPROJ_BASE', W['W_OUTPROJ'][0]),
         ('C_W_NORM_BASE', C['C_W_NORM'][0]), ('C_W_GNORM_BASE', C['C_W_GNORM'][0]), ('C_B_CONV_BASE', C['C_B_CONV'][0]),
         ('C_DT_BIAS_BASE', C['C_DT_BIAS'][0]), ('C_A_BASE', C['C_A'][0]), ('C_D_BASE', C['C_D'][0])]
    return d


def render_define(name, val):
    if name in ('H_ADDR_W', 'H_DEPTH', 'PT_MAIN_DEPTH'):
        return f'`define {name:<18} {val}'
    return f"`define {name:<18} 15'd{val}"


def write_vh(layouts):
    out = ['// Generated by scripts/m2_layout.py --write, do not edit.',
           '// Layout for d_state = 16*S; select S with +define+M2_S2, M2_S4 or M2_S8 (default S = 1).', '']
    first = True
    for S in (8, 4, 2):
        out.append(('`ifdef' if first else '`elsif') + f' M2_S{S}')
        first = False
        out += [render_define(n, v) for n, v in defines(layouts[S])]
    out.append('`else')
    out += [render_define(n, v) for n, v in defines(layouts[1])]
    out.append('`endif')
    open(LAYOUT_VH, 'w', newline='\n').write('\n'.join(out) + '\n')
    print(f'wrote {LAYOUT_VH}')


def check_rtl(layouts):
    """The S = 1 branch of m2_layout.vh must equal the script."""
    vh = open(LAYOUT_VH).read()
    last = vh[vh.rindex('`else'):]
    got = {m.group(1): int(m.group(2)) for m in (re.match(r"`define\s+(\w+)\s+(?:\d+'d)?(\d+)", l) for l in last.split('\n')) if m}
    want = dict(defines(layouts[1]))
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    assert not bad, f'm2_layout.vh S=1 branch disagrees with the script (file, script): {bad}'
    for S in (2, 4, 8):
        assert f'M2_S{S}' in vh
    print(f'm2_layout.vh matches the script (S=1: {len(want)} constants; S=2,4,8 present)')


def main_():
    layouts = {S: layout(S) for S in (1, 2, 4, 8)}
    for S, R in layouts.items():
        b0, b4 = R['blocks'][0], R['blocks'][4]
        print(f'S={S}: inp {b0["inp_g"]}/{b4["inp_g"]} xbc {b0["xbc_g"]}/{b4["xbc_g"]} (blocks 0-3 / 4); '
              f'mixer area {R["MIX_END"]} of bulk {R["BULK"]}, main depth {R["MAIN_DEPTH"]}; '
              f'weights {R["wcur"]} of {R["W_FC"]}; H depth {R["H_DEPTH"]}')
    if '--write' in sys.argv:
        write_vh(layouts)
    check_rtl(layouts)
    R = layouts[1]
    print('S=1 detail:')
    for k, (s, z) in R['main'].items():
        print(f'  {k:<11} {s:>4} .. {s + z:>4}  ({z} words)')
    print('LAYOUT OK')


if __name__ == '__main__':
    main_()
