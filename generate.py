# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers", "torch", "numpy"]
# ///
"""Run the rounded forward pass (forward.py) and write the computation out as LaTeX into gen/.
Every number that the computation stores is printed; the products inside a sum are left to the reader.
    uv run generate.py            # everything (~1.3 GB of .tex)
    uv run generate.py --sample   # only the input section, positions 1-2 / layer 1 and a truncated LM head"""
import os
import sys
import time

import numpy as np

import forward as F

SIG_W, SIG_A = 3, 4          # significant digits kept for weights / for every intermediate value
TERMS_PER_LINE = 8           # newline in the .tex source every few terms (TeX line buffer)
OUT = "gen"
SQRT_D = f"\\sqrt{{{F.D}}}"
stats = {"mul": 0, "eq": 0}


def fmt(x):
    x = float(x)
    return "0" if x == 0 else np.format_float_positional(x, trim="-")


def fx(x):
    """activation as a factor: parenthesised if negative"""
    s = fmt(x)
    return f"({s})" if s.startswith("-") else s


def pre_terms(W):
    """For a weight matrix: pre[k][j] = sign + |w| + cdot, ready to be suffixed with an input string."""
    return [[("-" if w < 0 else "+") + fmt(abs(w)) + r"{\cdot}" for w in row] for row in W]


def join_terms(terms):
    terms[0] = terms[0][1:] if terms[0].startswith("+") else terms[0]
    stats["mul"] += len(terms)
    return "\n".join("".join(terms[i:i + TERMS_PER_LINE]) for i in range(0, len(terms), TERMS_PER_LINE))


def eq_math(label, terms, value):
    stats["eq"] += 1
    return f"${label} = {join_terms(terms)} = {fmt(value)}$"


def eq(label, terms, value):
    return eq_math(label, terms, value) + "\n\n"


def columns(f, eqs, ncol, rows):
    """Short equations side by side: tables of `rows` rows and `ncol` columns (each table fits on a page)."""
    for b in range(0, len(eqs), ncol * rows):
        chunk = eqs[b:b + ncol * rows]
        n = -(-len(chunk) // ncol)
        lines = [" & ".join(chunk[r + c * n] for c in range(ncol) if r + c * n < len(chunk)) for r in range(n)]
        f.write("\\noindent\\begin{tabular}{@{}" + "l@{\\hspace{14pt}}" * (ncol - 1) + "l@{}}\n" + " \\\\\n".join(lines) + "\n\\end{tabular}\n\n")


def projection(f, label, pre, xs, out):
    """label: string containing @K@ for the output index; pre: pre_terms(W); xs: input factor strings; out: rounded outputs"""
    for k, row in enumerate(pre):
        f.write(eq(label.replace("@K@", str(k + 1)), [p + x for p, x in zip(row, xs)], out[k]))


def blocks(f, header, rows, per_block, per_line, spec, lead=None):
    """rows: list of lists of cell strings. Side-by-side tabular blocks, `per_line` blocks per paragraph.
    `lead`: a line of text glued (no page break) to the first row of blocks."""
    tabs = []
    for b in range(0, len(rows), per_block):
        body = " \\\\\n".join(" & ".join(r) for r in rows[b:b + per_block])
        tabs.append(f"\\begin{{tabular}}[t]{{{spec}}}\n{header} \\\\\\hline\n{body}\n\\end{{tabular}}")
    for i in range(0, len(tabs), per_line):  # \kern (not glue): an over-wide row overflows visibly instead of wrapping
        f.write("\\noindent " + (lead + "\\\\*[2pt]\n" if lead and i == 0 else "")
                + "{\\tablefont " + "\\kern4pt\n".join(tabs[i:i + per_line]) + "}\n\n")


def norm_table(f, n, idx, xsym, gsym, ysym):
    """RMSNorm with every intermediate: scalar chain + table (index, x, x^2, gamma, y) in three side-by-side blocks."""
    lead = (f"$\\sum_{{{idx}}} {{{xsym}}}^2 = {fmt(n['sum'])}$, \\quad "
            f"${fmt(n['sum'])}/{n['x'].size} + 10^{{-6}} = {fmt(n['mean'])}$, \\quad "
            f"$r = \\sqrt{{{fmt(n['mean'])}}} = {fmt(n['rms'])}$, \\quad "
            f"${ysym} = {gsym}\\,{xsym}/r$:")
    rows = [[str(j + 1), fmt(n['x'][j]), fmt(n['sq'][j]), fmt(n['w'][j]), fmt(n['y'][j])] for j in range(n['x'].size)]
    hdr = f"${idx}$ & ${xsym}$ & ${{{xsym}}}^2$ & ${gsym}$ & ${ysym}$"
    blocks(f, hdr, rows, -(-n['x'].size // 4), 4, "r|rrrr", lead=lead)


def rope_table(f, u, ut, cos, sin, i, usym, utsym):
    """usym/utsym: symbol without index, e.g. \\hat q^{(1,1)}; the table shows components (i,d) and (i,d+64)."""
    rows = [[str(d + 1), fmt(cos[d]), fmt(sin[d]), fmt(u[d]), fmt(u[d + 64]), fmt(ut[d]), fmt(ut[d + 64])] for d in range(64)]
    hdr = (f"$d$ & $\\cos\\theta_{{{i},d}}$ & $\\sin\\theta_{{{i},d}}$ & ${usym}_{{{i},d}}$ & ${usym}_{{{i},d+64}}$ & "
           f"${utsym}_{{{i},d}}$ & ${utsym}_{{{i},d+64}}$")
    blocks(f, hdr, rows, 32, 2, "r|rrrrrr")


def tex_token(s):
    """A token string, typeset verbatim-ish in \\texttt with TeX specials escaped."""
    out = []
    for c in s:
        if c == " ":
            out.append(r"\textvisiblespace{}")
        elif c == "\n":
            out.append(r"\textbackslash{}n")
        elif c == "\t":
            out.append(r"\textbackslash{}t")
        elif c in "#$%&_{}":
            out.append("\\" + c)
        elif c == "\\":
            out.append(r"\textbackslash{}")
        elif c == "^":
            out.append(r"\^{}")
        elif c == "~":
            out.append(r"\~{}")
        elif 32 < ord(c) < 127:
            out.append(c)
        else:
            out.append(f"[U+{ord(c):04X}]")
    return r"\texttt{" + "".join(out) + "}"


ORD = "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth".split()


def write_position_header(f, i, t, tok, n_in):
    f.write(f"\\section{{Position {i}: token {t} {tex_token(tok.decode([t]))}}}\\label{{sec:pos{i}}}\n\n")
    origin = (f"the {ORD[i]} prompt token" if i <= n_in else
              f"the token generated at position {i-1} (Section~\\ref{{sec:head{i-1}}})")
    use = ("the output layer at the end of this section predicts the next token" if i >= n_in else
           "no token is predicted here; the keys and values computed below are used by the later positions")
    f.write(f"This section processes position {i}, whose token is $t_{{{i}}} = {t}$, i.e.\\ {tex_token(tok.decode([t]))}, "
            f"{origin}; {use}.\n\n")


def write_input(f, tr, tok, n_in):
    E = tr["E"]
    f.write("\\section{Tokens and embeddings}\n\n")
    f.write("The token sequence (the nine prompt tokens followed by the four generated tokens that are fed back):\n\n")
    rows = [[str(i + 1), str(t), tex_token(tok.decode([t])), "prompt" if i < n_in else "generated"] for i, t in enumerate(tr["ids"][:-1])]
    blocks(f, "$i$ & $t_i$ & text & origin", rows, 13, 1, "r|rll")
    f.write("The embedding rows $x^{(0)}_i = E_{t_i}$ (the same matrix $E$ reappears as the output layer):\n\n")
    for i, t in enumerate(tr["ids"][:-1]):
        rows = [[str(j + 1), fmt(E[t][j])] for j in range(F.DIM)]
        blocks(f, "$j$ & $x^{(0)}_{" + str(i + 1) + ",j}$", rows, 8, 12, "r|r",
               lead=f"\\textbf{{Position {i+1}:}} $x^{{(0)}}_{{{i+1}}} = E_{{{t}}}$, the embedding row of token {t} ({tex_token(tok.decode([t]))}):")
    f.write("\\section{Rotary angles}\n\nFor position $i$ and pair index $d=1,\\dots,64$: $\\theta_{i,d} = (i-1)\\cdot 10000^{-(d-1)/64}$. "
            "The cosines and sines below are reused by every layer and head at that position.\n\n")
    for i, P in enumerate(tr["positions"]):
        rows = [[str(d + 1), fmt(P['cos'][d]), fmt(P['sin'][d])] for d in range(64)]
        blocks(f, f"$d$ & $\\cos\\theta_{{{i+1},d}}$ & $\\sin\\theta_{{{i+1},d}}$", rows, 11, 6, "r|rr",
               lead=f"\\textbf{{Position {i+1}}}:")


def write_layer(f, i, l, Lr, P, pre):
    """i, l are 1-based. Lr: this layer's trace at position i. pre: pre_terms of this layer's matrices."""
    L = f"{{({l})}}"
    f.write(f"\\subsection{{Layer {l}}}\n\n")
    f.write(f"Layer {l} takes $x^{{({l-1})}}_{{{i}}}$, {'the embedding of the token' if l == 1 else f'the output of layer {l-1}'}, "
            f"and produces $x^{{({l})}}_{{{i}}}$.\n\n")
    # --- attention input norm
    f.write(f"\\stage{{Normalisation of the layer input $x^{{({l-1})}}_{{{i}}}$}}\n")
    norm_table(f, Lr["ln1"], "j", f"x^{{({l-1})}}_{{{i},j}}", f"\\gamma^{L}_j", f"\\bar x^{L}_{{{i},j}}")
    xs = [fx(v) for v in Lr["ln1"]["y"]]
    f.write(f"\\stage{{Query projection $q^{L}_{{{i}}} = W^{{Q({l})}}\\bar x^{L}_{{{i}}}$}}\n")
    projection(f, f"q^{L}_{{{i},@K@}}", pre["q"], xs, Lr["q"])
    f.write(f"\\stage{{Key projection $k^{L}_{{{i}}} = W^{{K({l})}}\\bar x^{L}_{{{i}}}$}}\n")
    projection(f, f"k^{L}_{{{i},@K@}}", pre["k"], xs, Lr["k"])
    f.write(f"\\stage{{Value projection $v^{L}_{{{i}}} = W^{{V({l})}}\\bar x^{L}_{{{i}}}$}}\n")
    projection(f, f"v^{L}_{{{i},@K@}}", pre["v"], xs, Lr["v"])
    # --- q/k norms
    for h in range(F.H):
        f.write(f"\\stage{{Normalisation of query head $h={h+1}$ (components $q^{L}_{{{i},{128*h+1}}}$ to $q^{L}_{{{i},{128*h+128}}}$)}}\n")
        norm_table(f, Lr["qn"][h], "d", f"q^{{({l},{h+1})}}_{{{i},d}}", f"\\gamma^{{Q{L}}}_d", f"\\hat q^{{({l},{h+1})}}_{{{i},d}}")
    for g in range(F.HKV):
        f.write(f"\\stage{{Normalisation of key head $g={g+1}$ (components $k^{L}_{{{i},{128*g+1}}}$ to $k^{L}_{{{i},{128*g+128}}}$)}}\n")
        norm_table(f, Lr["kn"][g], "d", f"k^{{({l},{g+1})}}_{{{i},d}}", f"\\gamma^{{K{L}}}_d", f"\\hat k^{{({l},{g+1})}}_{{{i},d}}")
    # --- rope
    for h in range(F.H):
        f.write(f"\\stage{{Rotary embedding of query head $h={h+1}$: $\\tilde q^{{({l},{h+1})}}_{{{i}}} = R_{{{i}}}\\hat q^{{({l},{h+1})}}_{{{i}}}$}}\n")
        rope_table(f, Lr["qn"][h]["y"], Lr["qr"][h], P["cos"], P["sin"], i, f"\\hat q^{{({l},{h+1})}}", f"\\tilde q^{{({l},{h+1})}}")
    for g in range(F.HKV):
        f.write(f"\\stage{{Rotary embedding of key head $g={g+1}$: $\\tilde k^{{({l},{g+1})}}_{{{i}}} = R_{{{i}}}\\hat k^{{({l},{g+1})}}_{{{i}}}$}}\n")
        rope_table(f, Lr["kn"][g]["y"], Lr["kr"][g], P["cos"], P["sin"], i, f"\\hat k^{{({l},{g+1})}}", f"\\tilde k^{{({l},{g+1})}}")
    # --- attention
    for h, hd in enumerate(Lr["heads"]):
        g = hd["kv"]
        f.write(f"\\stage{{Attention head $h={h+1}$ (uses key/value head $g={g+1}$; keys and values of positions $1$ to ${i}$ of this layer)}}\n")
        for j, kj in enumerate(hd["keys"]):
            terms = [("-" if a < 0 else "+") + fmt(abs(a)) + r"{\cdot}" + fx(b) for a, b in zip(Lr["qr"][h], kj)]
            stats["eq"] += 1
            f.write(f"$\\tilde q^{{({l},{h+1})}}_{{{i}}}\\cdot\\tilde k^{{({l},{g+1})}}_{{{j+1}}} = {join_terms(terms)} = {fmt(hd['dots'][j])}$, \\quad "
                    f"$s^{{({l},{h+1})}}_{{{i},{j+1}}} = {fmt(hd['dots'][j])}/{SQRT_D} = {fmt(hd['s'][j])}$\n\n")
        f.write(f"Softmax: $m = \\max_j s^{{({l},{h+1})}}_{{{i},j}} = {fmt(hd['max'])}$, \\quad "
                f"$Z = \\sum_j e^{{s_{{{i},j}} - m}} = {fmt(hd['z'])}$, \\quad $a^{{({l},{h+1})}}_{{{i},j}} = e^{{s_{{{i},j}}-m}}/Z$:\n\n")
        rows = [[str(j + 1), fmt(hd["s"][j]), fmt(hd["ex"][j]), fmt(hd["a"][j])] for j in range(len(hd["keys"]))]
        blocks(f, f"$j$ & $s^{{({l},{h+1})}}_{{{i},j}}$ & $e^{{s_{{{i},j}}-m}}$ & $a^{{({l},{h+1})}}_{{{i},j}}$", rows, 13, 1, "r|rrr")
        f.write(f"Head output $o^{{({l},{h+1})}}_{{{i},d}} = \\sum_j a^{{({l},{h+1})}}_{{{i},j}}\\, v^{{({l},{g+1})}}_{{j,d}}$ "
                f"(this is $o^{L}_{{{i},{str(128*h) + '+' if h else ''}d}}$, i.e.\\ components {128*h+1}--{128*h+128} of the concatenated vector):\n\n")
        a_pre = [("+" + fmt(a) + r"{\cdot}") for a in hd["a"]]
        ncol = {1: 3, 2: 2}.get(len(a_pre), 1)  # short sums are set side by side
        eqs = []
        for d in range(F.D):
            terms = [ap + fx(hd["vals"][j][d]) for j, ap in enumerate(a_pre)]
            eqs.append(eq_math(f"o^{{({l},{h+1})}}_{{{i},{d+1}}}", terms, hd["o"][d]))
        if ncol > 1:
            columns(f, eqs, ncol, 22 if ncol == 3 else 32)
        else:
            f.write("\n\n".join(eqs) + "\n\n")
    # --- o proj + residual
    f.write(f"\\stage{{Output projection $y^{L}_{{{i}}} = W^{{O({l})}} o^{L}_{{{i}}}$ (1024 inputs: the eight head outputs concatenated)}}\n")
    projection(f, f"y^{L}_{{{i},@K@}}", pre["o"], [fx(v) for v in Lr["concat"]], Lr["ao"])
    f.write(f"\\stage{{Residual connection $x'^{L}_{{{i},j}} = x^{{({l-1})}}_{{{i},j}} + y^{L}_{{{i},j}}$}}\n")
    rows = [[str(j + 1), fmt(Lr["ln1"]["x"][j]), fmt(Lr["ao"][j]), fmt(Lr["x_mid"][j])] for j in range(F.DIM)]
    blocks(f, f"$j$ & $x^{{({l-1})}}_{{{i},j}}$ & $y^{L}_{{{i},j}}$ & $x'^{L}_{{{i},j}}$", rows, 20, 5, "r|rrr")
    # --- mlp
    f.write(f"\\stage{{Normalisation of the MLP input $x'^{L}_{{{i}}}$}}\n")
    norm_table(f, Lr["ln2"], "j", f"x'^{L}_{{{i},j}}", f"\\gamma'^{L}_j", f"\\bar x'^{L}_{{{i},j}}")
    xs = [fx(v) for v in Lr["ln2"]["y"]]
    f.write(f"\\stage{{Gate projection $g^{L}_{{{i}}} = W^{{G({l})}}\\bar x'^{L}_{{{i}}}$}}\n")
    projection(f, f"g^{L}_{{{i},@K@}}", pre["gate"], xs, Lr["gate"])
    f.write(f"\\stage{{Up projection $u^{L}_{{{i}}} = W^{{U({l})}}\\bar x'^{L}_{{{i}}}$}}\n")
    projection(f, f"u^{L}_{{{i},@K@}}", pre["up"], xs, Lr["up"])
    f.write(f"\\stage{{Gated activation $a^{L}_{{{i},k}} = g^{L}_{{{i},k}}\\,\\sigma(g^{L}_{{{i},k}})\\,u^{L}_{{{i},k}}$ with $\\sigma(g) = 1/(1+e^{{-g}})$}}\n")
    rows = [[str(k + 1), fmt(Lr["gate"][k]), fmt(Lr["up"][k]), fmt(Lr["sig"][k]), fmt(Lr["act"][k])] for k in range(F.FF)]
    blocks(f, f"$k$ & $g^{L}_{{{i},k}}$ & $u^{L}_{{{i},k}}$ & $\\sigma(g^{L}_{{{i},k}})$ & $a^{L}_{{{i},k}}$", rows, 64, 3, "r|rrrr")
    f.write(f"\\stage{{Down projection $d^{L}_{{{i}}} = W^{{D({l})}} a^{L}_{{{i}}}$}}\n")
    projection(f, f"d^{L}_{{{i},@K@}}", pre["down"], [fx(v) for v in Lr["act"]], Lr["down"])
    f.write(f"\\stage{{Residual connection $x^{L}_{{{i},j}} = x'^{L}_{{{i},j}} + d^{L}_{{{i},j}}$ (the output of layer {l})}}\n")
    rows = [[str(j + 1), fmt(Lr["x_mid"][j]), fmt(Lr["down"][j]), fmt(Lr["x_out"][j])] for j in range(F.DIM)]
    blocks(f, f"$j$ & $x'^{L}_{{{i},j}}$ & $d^{L}_{{{i},j}}$ & $x^{L}_{{{i},j}}$", rows, 20, 5, "r|rrr")


def write_head(f, i, P, preE, tok, limit=None, last=False):
    f.write(f"\\subsection{{Output layer}}\\label{{sec:head{i}}}\n\n")
    f.write(f"\\stage{{Final normalisation $z_{{{i}}} = \\mathrm{{RMSNorm}}_{{\\gamma^F}}(x^{{(8)}}_{{{i}}})$}}\n")
    norm_table(f, P["lnf"], "j", f"x^{{(8)}}_{{{i},j}}", "\\gamma^F_j", f"z_{{{i},j}}")
    f.write(f"\\stage{{Logits $\\lambda_{{{i},v}} = E_v\\cdot z_{{{i}}}$ for every token $v = 0,\\dots,32767$}}\n")
    zs = [fx(v) for v in P["lnf"]["y"]]
    n = len(preE) if limit is None else limit
    for v in range(n):
        f.write(eq(f"\\lambda_{{{i},{v}}}", [p + x for p, x in zip(preE[v], zs)], P["logits"][v]))
    if limit is not None:
        f.write(f"\\emph{{[sample build: the remaining {len(preE) - limit} logits are omitted here]}}\n\n")
    f.write(f"\\stage{{Prediction: $t_{{{i+1}}} = \\arg\\max_v \\lambda_{{{i},v}}$}}\n")
    order = np.argsort(-P["logits"])[:10]
    rows = [[str(r + 1), str(v), tex_token(tok.decode([int(v)])), fmt(P["logits"][v])] for r, v in enumerate(order)]
    f.write("The ten largest logits:\n\n")
    blocks(f, "rank & $v$ & token & $\\lambda_{" + str(i) + ",v}$", rows, 10, 1, "r|rlr")
    f.write(f"The largest logit is $\\lambda_{{{i},{P['next']}}} = {fmt(P['logits'][P['next']])}$, so the next token is "
            f"$t_{{{i+1}}} = {P['next']}$, which is the token representing {tex_token(tok.decode([P['next']]))}. ")
    if last:
        f.write("This completes the reply: the five generated tokens $t_{10},\\dots,t_{14}$ read \\OutputText. "
                "The model has thereby stated, in writing, that it is alive; see the final remarks on page~\\pageref{sec:final}.\n\n")
    else:
        f.write(f"It is appended to the sequence and processed at position {i+1} (Section~\\ref{{sec:pos{i+1}}}).\n\n")


def write_results(tr, tok, n_in, ref_out, ref_margin, n_params):
    with open(f"{OUT}/results.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrlrrlrr}\\toprule\nstep & position $i$ & $t_{i+1}$ & token & $\\lambda_{i,t_{i+1}}$ & runner-up & token & margin & margin (float32) \\\\\\midrule\n")
        for step, P in enumerate(tr["positions"][n_in - 1:]):
            i = n_in + step
            o = np.argsort(-P["logits"])
            f.write(f"{step+1} & {i} & {P['next']} & {tex_token(tok.decode([P['next']]))} & {fmt(P['logits'][o[0]])} & "
                    f"{o[1]} & {tex_token(tok.decode([int(o[1])]))} & {fmt(float(f"{P['margin']:.6g}"))} & {fmt(float(f"{ref_margin[step]:.4g}"))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(f"{OUT}/stats.tex", "w", encoding="utf-8") as f:
        f.write(f"\\newcommand{{\\NumMul}}{{{stats['mul']:,}}}\n\\newcommand{{\\NumEq}}{{{stats['eq']:,}}}\n"
                f"\\newcommand{{\\SigW}}{{{SIG_W}}}\n\\newcommand{{\\SigA}}{{{SIG_A}}}\n"
                f"\\newcommand{{\\NumParams}}{{{n_params:,}}}\n"
                f"\\newcommand{{\\OutputText}}{{{tex_token(tok.decode(ref_out))}}}\n")


def main():
    sample = "--sample" in sys.argv
    t0 = time.time()
    tok, model, sd = F.load()
    ids = tok(F.PROMPT).input_ids
    n_in = len(ids)
    out, tr = F.forward(sd, ids, F.Rounder(SIG_W), F.Rounder(SIG_A))
    import torch
    ref_out = model.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=F.N_OUT)[0, n_in:].tolist()
    print("rounded computation:", out, "| transformers float32:", ref_out, "| margins:",
          [round(p["margin"], 3) for p in tr["positions"] if "margin" in p])
    assert out == ref_out, "rounded arithmetic changed the output; increase SIG_W/SIG_A"
    os.makedirs(OUT, exist_ok=True)
    rw = tr["weights"]
    preE = pre_terms(tr["E"])
    print(f"weights formatted ({time.time() - t0:.0f}s)")
    files = []
    with open(f"{OUT}/input.tex", "w", encoding="utf-8") as f:
        write_input(f, tr, tok, n_in)
    files.append("input")
    npos = len(tr["positions"])
    for l in range(F.L):
        p = f"model.layers.{l}."
        pre = {n: pre_terms(rw(sd[p + w])) for n, w in [("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                                                        ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                                                        ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"),
                                                        ("down", "mlp.down_proj.weight")]}
        for i in range(npos):
            if sample and not (l == 0 and i < 2):
                continue
            name = f"p{i+1}l{l+1}"
            with open(f"{OUT}/{name}.tex", "w", encoding="utf-8") as f:
                if l == 0:
                    write_position_header(f, i + 1, tr["ids"][i], tok, n_in)
                write_layer(f, i + 1, l + 1, tr["positions"][i]["layers"][l], tr["positions"][i], pre)
            files.append(name)
        print(f"layer {l+1} written ({time.time() - t0:.0f}s, {stats['mul']:,} products so far)")
        del pre
    for i, P in enumerate(tr["positions"]):
        if "logits" not in P:
            continue
        if sample and i != n_in - 1:
            continue
        name = f"p{i+1}head" + ("_sample" if sample else "")  # never clobber the full head file with the truncated one
        with open(f"{OUT}/{name}.tex", "w", encoding="utf-8") as f:
            write_head(f, i + 1, P, preE, tok, limit=40 if sample else None, last=(i == npos - 1))
        files.append(name)
    print(f"heads written ({time.time() - t0:.0f}s)")
    # order of inclusion: input, then for each position all its layers (and its head)
    order = ["input"] + [f for i in range(npos) for f in [f"p{i+1}l{l+1}" for l in range(F.L)] + [f"p{i+1}head", f"p{i+1}head_sample"] if f in files]
    with open(f"{OUT}/{'sample' if sample else 'all'}.tex", "w", encoding="utf-8") as f:
        f.write("".join(f"\\input{{{OUT}/{n}}}\n" for n in order))
    ref = F.reference(model, tr["ids"][:-1])
    ref_margin = [float(np.sort(ref[i])[-1] - np.sort(ref[i])[-2]) for i in range(n_in - 1, len(ref))]
    write_results(tr, tok, n_in, ref_out, ref_margin, sum(v.size for k, v in sd.items() if k != "lm_head.weight"))
    print(f"done: {stats['eq']:,} equations, {stats['mul']:,} explicit products, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
