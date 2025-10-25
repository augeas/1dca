

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        ##Take Your Brain to the 1st Dimension
        ###(In which we piss about with reasonably elementary One-Dimensional Cellular Automata.)

        Consider a 1-dimensional strip of cells that can be in two states, live or dead, on or off. The new state of a cell is determined by its current state and those of its neighbours, with evolution in time represented by filling in sucessive rows down a grid. Typically, the central cell at the top of the grid is seeded as live. If only the left, (`l`) centre (`c`) and right (`r`) cells are considered there are $2^{3}=8$ combinations of states, and $2^{8}=256$ possible rules. These are [elementary cellular automata](https://en.wikipedia.org/wiki/Elementary_cellular_automaton). If the three states are considered as bits of the integers 0-7, the new state for each combination can be represented by an 8-bit integer, a scheme attributed to [Wolfram](https://tinyurl.com/wolframsacrank). If next-nearest neighbours are considered, left-of-left (`L`) and right-of-right (`R`), there are $2^{5}=32$ combinations of states, and $2^{32}=4294967296$ rules. An exhaustive search for the suposedly interesting ones would be rather tedious. [Toffoli and Margolus](https://people.csail.mit.edu/nhm/cam-book.pdf) developed dedicated hardware that enabled the programatic generation of rules in [Forth](https://en.wikipedia.org/wiki/Forth_(programming_language)). Somewhat inspired by this, here rules can be specified with simple (mostly) boolean expressions with an additional if-then-else function. (Done by converting the infix expression to [reverse-Polish](https://en.wikipedia.org/wiki/Reverse_Polish_notation) with a cheap implementation of the [shunting algortithm](https://en.wikipedia.org/wiki/Shunting_yard_algorithm). A look-up-table is then generated for all combinations of states.) If the previous state of the central cell (`p`) is considered, there are $2^{64}$ rules, which is really rather a lot.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from functools import partial, reduce
    import string

    from matplotlib import colormaps
    import numpy as np
    from PIL import Image
    return Image, colormaps, np, partial, reduce, string


@app.cell
def _():
    return


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
                    pred = np.ones((32,)) * stack.pop()
                    pred_true = np.ones((32,)) * stack.pop()
                    pred_false = np.ones((32,)) * stack.pop()
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
    

    def ca_seq(rule, seed, height, history=0, palette='YlGn', frames=1):
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
    return (ca_seq,)


@app.cell
def _(Image, build_rule, ca_seq, np):
    def ca1d_img(expr, width, height, history=0, palette='YlGn', max_dim=512):
        rule, err = build_rule(expr)
        if not err is None:
            return (Image.fromarray(np.zeros((max_dim, max_dim, 3), np.uint8)), err)
        cells = np.zeros((width, 1), np.int32)
        cells[width//2] = 1
        img = ca_seq(rule, cells, height, history, palette).__next__()
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
    return colour_box, height_dropdown, hist_dropdown, rule_box, width_dropdown


@app.cell
def _(
    ca1d_img,
    colour_box,
    height_dropdown,
    hist_dropdown,
    rule_box,
    width_dropdown,
):
    ui_img, rule_err = ca1d_img(rule_box.value, width_dropdown.value, height_dropdown.value,
        history=hist_dropdown.value, palette=colour_box.value
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
        * boolean operators: `&`, (and) `|`, or `^`, (xor) `<`, `=`, `>`
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
    mo.vstack([mo.plain_text('Some example rules:'), mo.hstack([
        eg_img('(L^c)|(r^c)', palette='plasma'),
        eg_img('((l+r)>(L+R+c))^(L&R)', palette='ocean'),
        eg_img('(l^L)|(R^(c&r))', palette='twilight'),
        eg_img('(l|L)^((L|R)^p)', palette='seismic'),
        eg_img('((l^R)|(R^p))', palette='turbo')
    ])])
    return


@app.cell
def _():
    return


@app.cell
def _():
    # (l^R)|(c&(L=r))
    # (l+L+c)>(r+R))^(R&r)
    # ((l+r)>(L+R+c))^(L&R)
    # (c+r+c*r+l*c*r)%2
    # (L|r)^(R&(c|l))
    # (l^L)|(R^(c&r))
    # (l|L)^((L|R)^p)
    # ((l^R)|(R^p))
    # ((l^p)|(L^c))
    # (L^c)|(r^c)
    # (l^L)|(c^R)
    # p+L+l+c+r+R+2
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
