

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        ##Take Your Brain to the 1st Dimension
        ###(In which we piss about with reasonably elementary One-Dimensional Cellular Automata.)

        Consider a 1-dimensional strip of cells that can be in two states, live or dead, on or off. The new state of a cell is determined by its current state and those of its neighbours, with evolution in time represented by filling in sucessive rows down a grid. Typically, the central cell at the top of the grid is seeded as live. If only the left, (`l`) centre (`c`) and right (`r`) cells are considered there are $2^{3}=8$ combinations of states, and $2^{8}=256$ possible rules. These are [elementary cellular automata](https://en.wikipedia.org/wiki/Elementary_cellular_automaton). If the three states are considered as bits of the integers 0-7, the new state for each combination can be represented by an 8-bit integer, a scheme attributed to [Wolfram](https://tinyurl.com/wolframsacrank). If next-nearest neighbours are considered, left-of-left (`L`) and right-of-right (`R`), there are $2^{5}=32$ combinations of states, and $2^{32}=4294967296$ rules. An exhaustive search for the suposedly interesting ones would be rather tedious. [Toffoli and Margolus](https://people.csail.mit.edu/nhm/cam-book.pdf) developed dedicated hardware that enabled the programatic generation of rules in [Forth](https://en.wikipedia.org/wiki/Forth_(programming_language)). Somewhat inspired by this, here rules can be specified with simple (mostly) boolean expressions with an additional if-then-else function. (Done by converting the infix expression to [reverse-Polish](https://en.wikipedia.org/wiki/Reverse_Polish_notation) with a cheap implementation of the [shunting algortithm](https://en.wikipedia.org/wiki/Shunting_yard_algorithm). A look-up-table is then generated for all combinations of states.) If the previous state of the central cell (`p`) is considered, there are $2^{2^{6}}$ rules, which is really rather a lot. By treating the previous states of each cell as a binary value, with the most recent value as the most-significant-bit, the cells can be coloured.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from collections import namedtuple
    from functools import partial, reduce
    from itertools import islice, groupby, starmap
    import string
    import time

    from matplotlib import colormaps
    import mido
    import numpy as np
    from PIL import Image
    return (
        Image,
        colormaps,
        islice,
        mido,
        namedtuple,
        np,
        partial,
        reduce,
        starmap,
        string,
        time,
    )


@app.cell
def _(np):
    __var_tokens__ = 'pLlcrR'
    __operators__ = {
        '*': 2, '%': 2,
        '+': 1, '-': 1,
        '&': 0, '|': 0, '^': 0, '~': 0, '=': 0, '<': 0, '>': 0
    }
    __op_tokens__ = ''.join(__operators__.keys())

    __op_funcs__ = {
        '*': np.multiply, '%': np.mod,
        '+': np.add, '-': np.subtract,
        '&': np.logical_and, '|': np.logical_or,
        '^': np.logical_xor, '~': np.logical_not,
        '=': np.equal,
        '<': np.less, '>': np.greater
    }

    __func_tokens__ = '?'
    return (
        __func_tokens__,
        __op_funcs__,
        __op_tokens__,
        __operators__,
        __var_tokens__,
    )


@app.cell
def _(__func_tokens__, __op_tokens__, __operators__, __var_tokens__, string):
    def shunt(expr):
        out_stack = list()
        op_stack = list()
        for i, tok in enumerate(expr):
            if tok in string.digits or tok in __var_tokens__:
                out_stack.append((tok, i))
            elif tok == '(' or tok in __func_tokens__:
                op_stack.append((tok, i))
            elif tok in __op_tokens__:
                priority = __operators__[tok]
                while len(op_stack):
                    if __operators__.get(op_stack[-1][0], -1) > priority:
                        out_stack.append(op_stack.pop())
                    else:
                        break
                op_stack.append((tok, i))
            elif tok == ')':
                if not len(op_stack):
                    return i
                while op_stack[-1][0] != '(':
                    try:
                        out_stack.append(op_stack.pop())
                    except:
                        return ([], i)
                op_stack.pop()
                if len(op_stack):
                    if op_stack[-1][0] in __func_tokens__:
                        out_stack.append(op_stack.pop())
            elif tok == ',':
                if len(op_stack):
                    while op_stack[-1][0] != '(':
                        try:
                            out_stack.append(op_stack.pop())
                        except:
                            return ([], i)
            else:
                return ([], i)
        while len(op_stack):
            out_stack.append(op_stack.pop())
        return (out_stack, None)
    return (shunt,)


@app.cell
def _(__op_funcs__, __op_tokens__, __var_tokens__, np, string):
    def rp_eval(expr, pos, vars):
        stack = list()
        for i, tok in enumerate(expr):
            if tok in string.digits:
                stack.append(np.int32(tok))
            elif tok in __var_tokens__:
                stack.append(vars[tok])
            elif tok in __op_tokens__:
                func = __op_funcs__[tok]
                if tok == '~':
                    try:
                        args = [stack.pop()]
                    except:
                        return (None, pos[i])
                else:
                    try:
                        arg2 = stack.pop()
                        arg1 = stack.pop()
                    except:
                        return (None, pos[i])
                    args = [arg1, arg2]
                stack.append(func(*args).astype(np.int32))
            elif tok == '?':
                try:
                    pred = np.ones((64,)) * stack.pop()
                    pred_true = np.ones((64,)) * stack.pop()
                    pred_false = np.ones((64,)) * stack.pop()
                except:
                    return (None, pos[i])
                stack.append(np.where(pred, pred_true, pred_false))
        return (stack[0].astype(np.uint32), None)
    return (rp_eval,)


@app.cell
def _(__var_tokens__, np, rp_eval, shunt):
    _all_states = np.arange(64)
    _all_bits = np.zeros((64, 6), np.bool)
    for i in range(6):
        _all_bits[:, i] = _all_states & 2**(5-i) > 0
    __vars__ = {var: _all_bits[:, i] for i, var in enumerate(__var_tokens__)}

    def build_rule(infix_rule):
        try:
            rp_toks, err = shunt(infix_rule)
        except:
            return ([], 0)
        if not err is None:
            return ([], err)
        toks, pos = zip(*rp_toks)
        rp_rule = ''.join(toks)
        return rp_eval(rp_rule, pos, __vars__)
    return (build_rule,)


@app.cell
def _(np, reduce):
    __shifts__ = ((16, -2), (8, -1), (2, 1), (1, 2))

    def ca_step(rule, cells):
        lsb = cells & 1
        state_bits = (factor * np.roll(lsb, shift) for factor, shift in __shifts__)
        states = reduce(np.add, state_bits, 4 * lsb) + 32 * ((cells >> 1) & 1)
        return rule[states]
    return (ca_step,)


@app.cell
def _(ca_step, colormaps, np, partial):
    def mono_render(cells):
        return 255 - 255 * (cells & 1)

    def palette_render(cells, cmap, mask):
        return (255 * cmap((mask - (cells & mask)) / mask)[:, 0, :-1]).astype(np.uint8)


    def ca_img_seq(rule, seed, height, history=0, palette='YlGn', frames=1):
        cells = np.ndarray.copy(seed)
        width, _ = cells.shape
        if not history:
            renderer = mono_render
        else:
            renderer = partial(palette_render,
                cmap=colormaps.get_cmap(palette),
                mask=2**(history+1) - 1
            )
        for _ in range(frames):
            img = np.zeros((height, width, 3), np.uint8)
            for i, row in enumerate(img):
                img[i, :, :] = renderer(cells)
                cells = (cells << 1) + (ca_step(rule, cells) & 1)
            yield img
    return (ca_img_seq,)


@app.cell
def _(Image, build_rule, ca_img_seq, np):
    def ca1d_img(expr, width, height, history=0, palette='YlGn', max_dim=512):
        rule, err = build_rule(expr)
        if not err is None:
            return (Image.fromarray(np.zeros((max_dim, max_dim, 3), np.uint8)), err)
        cells = np.zeros((width, 1), np.int32)
        cells[width//2] = 1
        img = ca_img_seq(rule, cells, height, history, palette).__next__()
        mag = max_dim // max(width, height)
        return (Image.fromarray(img).resize((mag*width, mag*height), 0), None)
    return (ca1d_img,)


@app.cell
def _():
    cmap_names = [
        'binary', 'gray', 'bone', 'seismic', 'vanimo', 'managua', 'berlin', 'Spectral',
        'twilight', 'twilight_shifted', 'ocean', 'turbo', 'plasma', 'magma', 'inferno', 'brg', 'gnuplot',
        'terrain', 'gist_earth'
    ]
    return (cmap_names,)


@app.cell
def _(cmap_names, colormaps, mo):
    rule_box = mo.ui.text('(l^R)|(c^L)', label='rule')
    hist_dropdown = mo.ui.dropdown(list(range(8)), value=4, label='history')
    dims = list(map(lambda n: 2**n, range(5, 10)))
    width_dropdown = mo.ui.dropdown(dims, value=512, label='width')
    height_dropdown = mo.ui.dropdown(dims, value=512, label='height')
    palettes = [name for name in cmap_names if name in colormaps]
    colour_box = mo.ui.dropdown(palettes, value='twilight', label='palette')
    return (
        colour_box,
        dims,
        height_dropdown,
        hist_dropdown,
        rule_box,
        width_dropdown,
    )


@app.cell
def _(
    ca1d_img,
    colour_box,
    dims,
    height_dropdown,
    hist_dropdown,
    rule_box,
    width_dropdown,
):
    ui_img, rule_err = ca1d_img(rule_box.value, width_dropdown.value, height_dropdown.value,
        history=hist_dropdown.value, palette=colour_box.value, max_dim=dims[-1]
    )
    return rule_err, ui_img


@app.cell
def _(rule_box, rule_err):
    if rule_err is None:
        err_txt = ''
    else:
        err_txt = '\n'.join([rule_box.value, ' '*rule_err+'^'])
    return (err_txt,)


@app.cell
def _(
    colour_box,
    err_txt,
    height_dropdown,
    hist_dropdown,
    mo,
    rule_box,
    ui_img,
    width_dropdown,
):
    mo.vstack([
        mo.hstack([rule_box, mo.plain_text(err_txt)]),
        mo.hstack([hist_dropdown, width_dropdown, height_dropdown, colour_box], justify='center'),
        ui_img,
    ], align='center')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ###Rule Syntax

        Each token in a rule is a single character:

        * variables: spatial, `L`, `l`, `c`, `r`, `R` and `p` (previous), 0 or 1
        * constants: single digits 0-9
        * integer operators: `+`, `-`, `*`, `%` (modular division)
        * boolean operators: `&`, (and) `|`, (or) `^`, (xor) `<`, `=`, `>`
        * parentheses: `()`
        * predicate function: `?` (`?(predicate, expression-if-true, expression-if-false)`)
        """
    )
    return


@app.cell
def _(ca1d_img, cmap_names, mo):
    def eg_img(rule, size=128, palette=cmap_names[0], history=3):
        return mo.vstack([
            mo.plain_text(rule),
            ca1d_img(rule, size, size, max_dim=size, palette=palette, history=history)[0]
        ], heights=[0,0])

    return (eg_img,)


@app.cell
def _(eg_img, mo):
    mo.vstack([
        mo.plain_text('Some example rules:'),
        mo.hstack([
            eg_img('(L^c)|(r^c)', palette='plasma'),
            eg_img('((l+r)>(L+R+c))^(L&R)', palette='ocean'),
            eg_img('(l|L)^((L|R)^p)', palette='seismic'),
            eg_img('((l^R)|(R^p))', palette='turbo')
        ]),
        mo.hstack([
            eg_img('((l^p)|(L^c))', palette='magma'),
            eg_img('(l^L)|(c^R)', palette='gray'),
            eg_img('l^(c|r)', palette='ocean'),
            eg_img('(c+r+c*r+l*c*r)%2', palette='brg')
        ])
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""(The last two are the moderately infamous [rule 30](https://en.wikipedia.org/wiki/Rule_30)) and [rule 110](https://en.wikipedia.org/wiki/Elementary_cellular_automaton).)""")
    return


@app.cell
def _(np):
    __midi_bits__ = np.array([16, 8, 4, 2, 1], dtype=np.uint32).reshape((5,1))

    __volca_beats__ = (42, 43, 36, 50, 46)
    __volca_keys__ = {
        'detune': 42, 'cutoff': 44, 'lfo': 46, 'pitchint': 47, 'cutoffint': 48
    }

    return __midi_bits__, __volca_beats__


@app.cell
def _(mido, mo):
    port_scan_button = mo.ui.button(
        label='refresh MIDI ports',
        value=mido.get_output_names(),
        on_click=lambda _: mido.get_output_names()
    )
    return (port_scan_button,)


@app.cell
def _(mo, port_scan_button):
    midi_port_box = mo.vstack([
        port_scan_button,
        mo.vstack([' '.join((str(i), name)) for i, name in enumerate(port_scan_button.value)])
    ])

    n_ports = len(port_scan_button.value)
    max_port = n_ports - 1
    return max_port, midi_port_box, n_ports


@app.cell
def _(max_port, mo, n_ports):
    def channel_boxes(n=5, default=None):
        return [mo.ui.number(value=default, start=0, stop=15, full_width=False) for _ in range(n)]

    def port_boxes(n=5):
        return [
            mo.ui.number(value=0, start=0, stop=n_ports, full_width=False) for _ in range(n)
        ]

    def notecc_boxes(defaults=(0, 0, 0 ,0 ,0)):
        return [mo.ui.number(value=df, start=0, stop=127, full_width=False) for df in defaults]

    def channel_box(default=-1):
        return mo.ui.number(value=default, start=-1, stop=15, full_width=False)

    def port_box(default=-1):
        return mo.ui.number(value=default, start=-1, stop=max_port, full_width=False)

    def notecc_box(default=0, mx=127):
        return mo.ui.number(value=default, start=0, stop=mx, full_width=False)
    return channel_box, notecc_box, port_box


@app.cell
def _(__volca_beats__, channel_box, notecc_box, port_box):
    dnb0 = notecc_box(__volca_beats__[0]); dnb1 = notecc_box(__volca_beats__[1]);
    dnb2 = notecc_box(__volca_beats__[2]); dnb3 = notecc_box(__volca_beats__[3]);
    dnb4 = notecc_box(__volca_beats__[4]);

    dpb0 = port_box(); dpb1 = port_box(); dpb2 = port_box(); dpb3 = port_box();
    dpb4 = port_box();

    dcb0 = channel_box(); dcb1 = channel_box(); dcb2 = channel_box(); dcb3 = channel_box();
    dcb4 = channel_box();
    return (
        dcb0,
        dcb1,
        dcb2,
        dcb3,
        dcb4,
        dnb0,
        dnb1,
        dnb2,
        dnb3,
        dnb4,
        dpb0,
        dpb1,
        dpb2,
        dpb3,
        dpb4,
    )


@app.cell
def _(
    dcb0,
    dcb1,
    dcb2,
    dcb3,
    dcb4,
    dnb0,
    dnb1,
    dnb2,
    dnb3,
    dnb4,
    dpb0,
    dpb1,
    dpb2,
    dpb3,
    dpb4,
    mo,
    n_ports,
):
    drum_gbl_port_box = mo.ui.number(value=1, start=0, stop=n_ports,
        label='global port', full_width=False)
    drum_gbl_chan_box = mo.ui.number(value=9, start=0, stop=15, label='global channel', full_width=False)

    drum_note_boxes = [dnb0, dnb1, dnb2, dnb3, dnb4]
    drum_channel_boxes = [dcb0, dcb1, dcb2, dcb3, dcb4]
    drum_port_boxes = [dpb0, dpb1, dpb2, dpb3, dpb4]

    gbl_bit_depth_box = mo.ui.number(value=3, start=1, stop=7, label='global bit depth')
    return (
        drum_channel_boxes,
        drum_gbl_chan_box,
        drum_gbl_port_box,
        drum_note_boxes,
        drum_port_boxes,
    )


@app.cell
def _(channel_box, notecc_box, port_box):
    vpb0 = port_box(); vpb1 = port_box(); vpb2 = port_box(); vpb3 = port_box(); vpb4 = port_box();

    vcb0 = channel_box(); vcb1 = channel_box(); vcb2 = channel_box(); vcb3 = channel_box(); vcb4 = channel_box();

    vob0 = notecc_box(); vob1 = notecc_box(); vob2 = notecc_box(); vob3 = notecc_box(); vob4 = notecc_box();

    vob0 = notecc_box(); vob1 = notecc_box(); vob2 = notecc_box(); vob3 = notecc_box(); vob4 = notecc_box();

    vccb0 = notecc_box(); vccb1 = notecc_box(); vccb2 = notecc_box(); vccb3 = notecc_box(); vccb4 = notecc_box();

    vcob0 = notecc_box(); vcob1 = notecc_box(); vcob2 = notecc_box(); vcob3 = notecc_box(); vcob4 = notecc_box();
    return (
        vcb0,
        vcb1,
        vcb2,
        vcb3,
        vcb4,
        vccb0,
        vccb1,
        vccb2,
        vccb3,
        vccb4,
        vcob0,
        vcob1,
        vcob2,
        vcob3,
        vcob4,
        vob0,
        vob1,
        vob2,
        vob3,
        vob4,
        vpb0,
        vpb1,
        vpb2,
        vpb3,
        vpb4,
    )


@app.cell
def _(
    mo,
    n_ports,
    vcb0,
    vcb1,
    vcb2,
    vcb3,
    vcb4,
    vccb0,
    vccb1,
    vccb2,
    vccb3,
    vccb4,
    vcob0,
    vcob1,
    vcob2,
    vcob3,
    vcob4,
    vob0,
    vob1,
    vob2,
    vob3,
    vob4,
    vpb0,
    vpb1,
    vpb2,
    vpb3,
    vpb4,
):
    vert_note_gbl_port_box = mo.ui.number(value=3, start=0, stop=n_ports,
        label='global port', full_width=False)
    vert_note_gbl_chan_box = mo.ui.number(value=None, start=0, stop=15, label='global channel', full_width=False)
    vert_note_gbl_offset_box = mo.ui.number(value=16, start=0, stop=95, label='global note offset', full_width=False)
    vert_note_gbl_cc_box = mo.ui.number(value=41, start=-1, stop=127, label='global CC', full_width=False)
    vert_note_gbl_cc_off_box = mo.ui.number(value=None, start=-1, stop=95, label='global CC offset')

    vert_note_port_boxes = [vpb0, vpb1, vpb2, vpb3, vpb4]
    vert_note_channel_boxes = [vcb0, vcb1, vcb2, vcb3, vcb4]
    vert_note_offset_boxes = [vob0, vob1, vob2, vob3, vob4]
    vert_note_cc_boxes = [vccb0, vccb1, vccb2, vccb3, vccb4]
    vert_note_cc_off_boxes = [vcob0, vcob1, vcob2, vcob3, vcob4]
    return (
        vert_note_cc_boxes,
        vert_note_cc_off_boxes,
        vert_note_channel_boxes,
        vert_note_gbl_cc_box,
        vert_note_gbl_cc_off_box,
        vert_note_gbl_chan_box,
        vert_note_gbl_offset_box,
        vert_note_gbl_port_box,
        vert_note_offset_boxes,
        vert_note_port_boxes,
    )


@app.cell
def _(notecc_box):
    hccb0 = notecc_box(42); hccb1 = notecc_box(44); hccb2 = notecc_box(46); hccb3 = notecc_box(47); hccb4 = notecc_box(48);

    hcob0 = notecc_box(32); hcob1 = notecc_box(8); hcob2 = notecc_box(8); hcob3 = notecc_box(32); hcob4 = notecc_box(32);
    return hccb0, hccb1, hccb2, hccb3, hccb4, hcob0, hcob1, hcob2, hcob3, hcob4


@app.cell
def _(
    hccb0,
    hccb1,
    hccb2,
    hccb3,
    hccb4,
    hcob0,
    hcob1,
    hcob2,
    hcob3,
    hcob4,
    mo,
    n_ports,
):
    horiz_note_port_box = mo.ui.number(value=2, start=-1, stop=n_ports,
        label='port', full_width=False)
    horiz_note_chan_box = mo.ui.number(value=None, start=0, stop=15, label='channel', full_width=False)
    horiz_note_offset_box = mo.ui.number(value=32, start=0, stop=95, label='note offset', full_width=False)

    horiz_note_cc_boxes = [hccb0, hccb1, hccb2, hccb3, hccb4]
    horiz_note_cc_off_boxes = [hcob0, hcob1, hcob2, hcob3, hcob4]
    return (
        horiz_note_cc_boxes,
        horiz_note_cc_off_boxes,
        horiz_note_chan_box,
        horiz_note_offset_box,
        horiz_note_port_box,
    )


@app.cell
def _(
    drum_channel_boxes,
    drum_gbl_chan_box,
    drum_gbl_port_box,
    drum_note_boxes,
    drum_port_boxes,
    mo,
):
    drum_boxes = mo.vstack([
        mo.hstack([drum_gbl_port_box, drum_gbl_chan_box]),
        mo.hstack([mo.vstack(list(drum)) for drum in zip(drum_note_boxes, drum_channel_boxes, drum_port_boxes)])
    ])
    return (drum_boxes,)


@app.cell
def _(
    mo,
    vert_note_cc_boxes,
    vert_note_cc_off_boxes,
    vert_note_channel_boxes,
    vert_note_gbl_cc_box,
    vert_note_gbl_cc_off_box,
    vert_note_gbl_chan_box,
    vert_note_gbl_offset_box,
    vert_note_gbl_port_box,
    vert_note_offset_boxes,
    vert_note_port_boxes,
):
    vert_boxes = mo.vstack([
        mo.hstack([
            vert_note_gbl_port_box, vert_note_gbl_chan_box, vert_note_gbl_offset_box,
            vert_note_gbl_cc_box, vert_note_gbl_cc_off_box
        ]),
        mo.hstack([mo.vstack(list(vert)) for vert in zip(
            ['ports'] + vert_note_port_boxes,
            ['channels'] + vert_note_channel_boxes,
            ['note offsets'] + vert_note_offset_boxes,
            ['CCs'] + vert_note_cc_boxes,
            ['CC offsets'] + vert_note_cc_off_boxes
        )])
    ])
    return (vert_boxes,)


@app.cell
def _(
    horiz_note_cc_boxes,
    horiz_note_cc_off_boxes,
    horiz_note_chan_box,
    horiz_note_offset_box,
    horiz_note_port_box,
    mo,
):
    horiz_boxes = mo.vstack([
        mo.hstack([horiz_note_port_box, horiz_note_chan_box, horiz_note_offset_box]),
        mo.hstack([mo.vstack(list(horiz)) for horiz in zip(
            ['CCs'] + horiz_note_cc_boxes, ['CC offsets'] + horiz_note_cc_off_boxes
        )])
    ])
    return (horiz_boxes,)


@app.cell
def _(midi_port_box):
    midi_port_box 
    return


@app.cell
def _(drum_boxes, mo):
    mo.vstack(['drums', drum_boxes])
    return


@app.cell
def _(mo, vert_boxes):
    mo.vstack(['vertical notes', vert_boxes])
    return


@app.cell
def _(horiz_boxes, mo):
    mo.vstack(['horizontal notes', horiz_boxes])
    return


@app.cell
def _(namedtuple):
    DRUM = namedtuple('DRUM', ['port', 'channel', 'note'])
    VERTICAL_NOTE = namedtuple('VERTICAL_NOTE', ['port', 'channel', 'note_offset', 'cc', 'cc_offset'])
    HORIZ_NOTE = namedtuple('HORIZONTAL_NOTE', ['port', 'channel', 'note_offset'])
    CC = namedtuple('CC', ['port', 'channel', 'cc', 'offset'])
    return CC, DRUM, HORIZ_NOTE, VERTICAL_NOTE


@app.function
def box_values(boxes, default=None):
    return [bx.value if bx.value >= 0 else default for bx in boxes]


@app.cell
def _(
    DRUM,
    drum_channel_boxes,
    drum_gbl_chan_box,
    drum_gbl_port_box,
    drum_note_boxes,
    drum_port_boxes,
    starmap,
):
    midi_drums = list(starmap(DRUM, zip(*starmap(box_values,
        (
            (drum_port_boxes, drum_gbl_port_box.value),
            (drum_channel_boxes, drum_gbl_chan_box.value),
            (drum_note_boxes, None)
        )
    ))))
    return (midi_drums,)


@app.cell
def _(
    VERTICAL_NOTE,
    starmap,
    vert_note_cc_boxes,
    vert_note_cc_off_boxes,
    vert_note_channel_boxes,
    vert_note_gbl_cc_box,
    vert_note_gbl_cc_off_box,
    vert_note_gbl_chan_box,
    vert_note_gbl_offset_box,
    vert_note_gbl_port_box,
    vert_note_offset_boxes,
    vert_note_port_boxes,
):
    vertical_midi_notes = list(starmap(VERTICAL_NOTE, zip(*starmap(box_values,
        (
            (vert_note_port_boxes, vert_note_gbl_port_box.value),
            (vert_note_channel_boxes, vert_note_gbl_chan_box.value),
            (vert_note_offset_boxes, vert_note_gbl_offset_box.value),
            (vert_note_cc_boxes, vert_note_gbl_cc_box.value),
            (vert_note_cc_off_boxes, vert_note_gbl_cc_off_box.value)
        )
    ))))
    return (vertical_midi_notes,)


@app.cell
def _(
    CC,
    HORIZ_NOTE,
    horiz_note_cc_boxes,
    horiz_note_cc_off_boxes,
    horiz_note_chan_box,
    horiz_note_offset_box,
    horiz_note_port_box,
    starmap,
):
    horiz_midi_note = HORIZ_NOTE(
        horiz_note_port_box.value,
        horiz_note_chan_box.value,
        horiz_note_offset_box.value    
    )

    horiz_midi_ccs = list(starmap(CC, zip(*map(box_values,
        (
            5 * (horiz_note_port_box,),
            5 * (horiz_note_chan_box,),
            horiz_note_cc_boxes,
            horiz_note_cc_off_boxes
        )
    ))))
    return horiz_midi_ccs, horiz_midi_note


@app.cell
def _(__midi_bits__, ca_step, np):
    def midi_synth_seq(
        rule, width, bit_depth, drum_notes, vert_notes, horiz_note, horiz_ccs):

        drum_keys = set([(note.port, note.channel) for note in drum_notes])
    
        vert_note_offsets = np.array([off.note_offset for off in vert_notes], dtype=np.int32)

        vert_note_ccs = {(note.port, note.channel): note.cc for note in vert_notes}
        vert_note_cc_offsets = {(note.port, note.channel): note.cc_offset for note in vert_notes}
        vert_note_keys = tuple((note.port, note.channel) for note in vert_notes)
        vert_note_uniq_keys = set(vert_note_keys)
    
        horiz_cc_offsets = np.array([cc.offset for cc in horiz_ccs])
        horiz_cc_nums = tuple(cc.cc for cc in horiz_ccs)
        horiz_key = (horiz_note.port, horiz_note.channel)

        bit_mask = 2**(bit_depth+1) - 1
    
        cells = np.zeros((width, 1), np.int32)
        centre = width // 2
        cells[centre] = 1
        centre_slice = slice(centre-2, centre+3)

        vert_note_values = {key: set() for key in vert_note_uniq_keys}
        h_note = None
    
        while True:
            cell_slice = cells[centre_slice] & 1
            drum_notes_on = {key: set() for key in drum_keys}
            for drum, cell in zip(drum_notes, cell_slice):
                if cell:
                    drum_notes_on[(drum.port, drum.channel)].add(drum.note)

            vert_strip = bit_mask - (cells[centre_slice].reshape((5,)) & bit_mask)
            horiz_strip = (__midi_bits__ * cell_slice).sum()

            vert_notes_on = {key: set() for key in vert_note_uniq_keys}
            vert_notes_off = {key: set() for key in vert_note_uniq_keys}
        
            for key, value, offset in zip(vert_note_keys, vert_strip, vert_note_offsets):
                if value and key[0] >= 0:
                    note_value = int(value + offset)
                    if note_value not in vert_note_values[key]:
                        vert_notes_on[key].add(note_value)

            if not any(vert_notes_on.values()):
                vert_notes_on = dict()
            if not any(vert_notes_off.values()):
                vert_notes_off = dict()
            
            vert_notes_off = {
                key: vert_note_values[key] - vert_notes_on.get(key, set()) for key in vert_note_keys
            }
            vert_note_values = vert_notes_on.copy()

            vert_ccs = {
                key: ((vert_note_ccs[key], int(vert_note_cc_offsets[key] + horiz_strip)),)
                    for key in vert_note_keys
            }
        
            new_note = int(horiz_note.note_offset + horiz_strip)
            if new_note == h_note:
                horiz_notes_on = dict()
                horiz_notes_off = dict()
            else:
                horiz_notes_on = {horiz_key: set((new_note,))}
                if h_note:
                    horiz_notes_off = {horiz_key: set((h_note,))}
                else:
                    horiz_notes_off = dict()
                h_note = new_note

            h_ccs = {
                horiz_key: tuple((num, val) for num, val in zip(
                    horiz_cc_nums, horiz_cc_offsets + vert_strip
                ))
            }
        
            all_notes_on = {**drum_notes_on, **vert_notes_on, **horiz_notes_on}
            all_notes_off = {**vert_notes_off, **horiz_notes_off}
            all_ccs = {**vert_ccs, **h_ccs}
        
            yield (all_notes_on, all_notes_off, all_ccs)

            cells = (cells << 1) + (ca_step(rule, cells) & 1)
                            
    return (midi_synth_seq,)


@app.cell
def _(dims, mo):
    bit_depth_dropdown = mo.ui.dropdown(list(range(8)), value=5, label='bit_depth')
    midi_width_dropdown = mo.ui.dropdown(dims, value=512, label='width')
    beats_dropdown = mo.ui.dropdown(
        [2**n for n in range(6,12)], value=256, label='beats'
    )
    duration_dropdown = mo.ui.dropdown(
        list(range(100, 1100, 50)), value=200, label='duration (ms)'
    )
    return (
        beats_dropdown,
        bit_depth_dropdown,
        duration_dropdown,
        midi_width_dropdown,
    )


@app.cell
def _(
    beats_dropdown,
    bit_depth_dropdown,
    build_rule,
    horiz_midi_ccs,
    horiz_midi_note,
    islice,
    midi_drums,
    midi_synth_seq,
    midi_width_dropdown,
    rule_box,
    vertical_midi_notes,
):
    def build_seq():
        midi_rule, midi_err = build_rule(rule_box.value)

        if midi_err is None:
            seq = midi_synth_seq(
                midi_rule, midi_width_dropdown.value, bit_depth_dropdown.value,
                midi_drums, vertical_midi_notes, horiz_midi_note, horiz_midi_ccs 
            )    
            yield from islice(seq, beats_dropdown.value)
    return (build_seq,)


@app.cell
def _(duration_dropdown, mido, time):
    def play_seq(seq):
        if not seq:
            return
        dt = duration_dropdown.value / 1000.0
        ports = list(map(mido.open_output, mido.get_output_names()))
        for notes_on, notes_off, controls in seq:
            for port_chan, notes in notes_off.items():
                port, chan = port_chan
                for note in notes:
                    ports[port].send(mido.Message('note_off', note=note, channel=chan))
            for port_chan, notes in notes_on.items():
                port, chan = port_chan
                for note in notes:
                    ports[port].send(mido.Message('note_on', note=note, channel=chan))
            for port_chan, ctrls in controls.items():
                port, chan = port_chan
                for ctrl in ctrls:
                    cc, val = ctrl
                    ports[port].send(
                        mido.Message('control_change', channel=chan, control=cc, value=val)
                    )
            time.sleep(dt)
        for port_chan, notes in notes_on.items():
            port, chan = port_chan
            for note in notes:
                ports[port].send(mido.Message('note_off', note=note, channel=chan))
    return (play_seq,)


@app.cell
def _(build_seq, mo):
    play_button = mo.ui.button(
        value=None, label='play',
        on_click=lambda seq: None if seq else build_seq()
    )
    return (play_button,)


@app.cell
def _(
    beats_dropdown,
    bit_depth_dropdown,
    duration_dropdown,
    midi_width_dropdown,
    mo,
    play_button,
):
    song_box = mo.hstack([
        play_button,
        bit_depth_dropdown,
        midi_width_dropdown,
        beats_dropdown,
        duration_dropdown
    ])
    return (song_box,)


@app.cell
def _(song_box):
    song_box
    return


@app.cell
def _(play_button, play_seq):
    play_seq(play_button.value)
    return


if __name__ == "__main__":
    app.run()
