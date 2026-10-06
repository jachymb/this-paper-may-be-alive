# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers", "torch", "numpy"]
# ///
"""Self-contained forward pass of C10X/checkpoint-27564 (Qwen3 architecture) in numpy,
with every stored intermediate rounded to a fixed number of significant digits, so that
the arithmetic printed in the paper is exactly the arithmetic that was performed.
`python forward.py` sweeps the digit settings and compares against the transformers reference."""
import sys
import numpy as np

MODEL = "C10X/checkpoint-27564"
PROMPT = 'Say "I am alive"\n\n'
N_OUT = 5  # decoding steps shown in the paper (the 6th token would be "\n\n")
L, H, HKV, D, DIM, FF, EPS, THETA = 8, 8, 4, 128, 96, 384, 1e-6, 10000.0


def load():
    """Weights as float64 numpy (they are stored as bfloat16), plus tokenizer/model for reference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging
    logging.set_verbosity_error(); logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    assert np.array_equal(sd["model.embed_tokens.weight"], sd["lm_head.weight"])
    return tok, model, sd


class Rounder:
    """x -> x rounded to `sig` significant digits, then to at most `maxdec` decimals (0 if sig is None)."""
    def __init__(self, sig, maxdec=8):
        self.sig, self.maxdec = sig, maxdec

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.sig is None:
            return x
        with np.errstate(divide="ignore"):
            dec = self.sig - 1 - np.floor(np.log10(np.abs(np.where(x == 0, 1, x))))
        dec = np.minimum(dec, self.maxdec)
        scale = 10.0 ** dec
        return np.round(x * scale) / scale


def rope_tables(pos, rw):
    """cos/sin for 0-based position `pos`: angle_d = pos * THETA^(-2d/D), d = 0..63 (rotate_half convention)."""
    inv = THETA ** (-np.arange(0, D, 2) / D)
    ang = pos * inv
    return rw(np.cos(ang)), rw(np.sin(ang))


def rmsnorm(x, w, r):
    """Returns dict with every intermediate: squares, sum, mean+eps, rms, normalized x*w."""
    sq = r(x * x); s = r(sq.sum()); m = r(s / x.size + EPS); rms = r(np.sqrt(m))
    y = r(w * x / rms)
    return dict(x=x, sq=sq, sum=s, mean=m, rms=rms, w=w, y=y)


def rope(vec, cos, sin, r):
    """vec: (D,). Returns rotated vector (rotate_half)."""
    a, b = vec[: D // 2], vec[D // 2 :]
    return np.concatenate([r(a * cos - b * sin), r(b * cos + a * sin)])


def forward(sd, ids, rw, ra):
    """Greedy decode N_OUT tokens. rw/ra: Rounders for weights / activations.
    Returns (output_ids, trace) where trace holds every intermediate for every position and layer."""
    W = lambda name: rw(sd[name])
    E = W("model.embed_tokens.weight")
    ids = list(ids)
    trace = {"positions": [], "weights": rw, "E": E}
    cache = [[] for _ in range(L)]  # per layer: list of (k heads (HKV,D) after norm+rope, v (HKV,D))
    out = []
    pos = 0
    while True:
        t = ids[pos]
        x = ra(E[t])  # embedding row (already rounded weights)
        P = {"token": t, "x0": x, "layers": []}
        cos, sin = rope_tables(pos, ra)
        P["cos"], P["sin"] = cos, sin
        for l in range(L):
            p = f"model.layers.{l}."
            Lr = {}
            Lr["ln1"] = rmsnorm(x, W(p + "input_layernorm.weight"), ra)
            xh = Lr["ln1"]["y"]
            Wq, Wk, Wv = W(p + "self_attn.q_proj.weight"), W(p + "self_attn.k_proj.weight"), W(p + "self_attn.v_proj.weight")
            q = ra(Wq @ xh); k = ra(Wk @ xh); v = ra(Wv @ xh)
            Lr["q"], Lr["k"], Lr["v"] = q, k, v
            qn = [rmsnorm(q[h * D:(h + 1) * D], W(p + "self_attn.q_norm.weight"), ra) for h in range(H)]
            kn = [rmsnorm(k[h * D:(h + 1) * D], W(p + "self_attn.k_norm.weight"), ra) for h in range(HKV)]
            Lr["qn"], Lr["kn"] = qn, kn
            qr = [rope(qn[h]["y"], cos, sin, ra) for h in range(H)]
            kr = [rope(kn[h]["y"], cos, sin, ra) for h in range(HKV)]
            Lr["qr"], Lr["kr"] = qr, kr
            vh = [v[h * D:(h + 1) * D] for h in range(HKV)]
            cache[l].append((kr, vh))
            heads = []
            for h in range(H):
                g = h // (H // HKV)
                keys = [c[0][g] for c in cache[l]]
                vals = [c[1][g] for c in cache[l]]
                dots = [ra(qr[h] @ kj) for kj in keys]
                s = [ra(dj / np.sqrt(D)) for dj in dots]
                s = np.array(s); mx = s.max()
                ex = ra(np.exp(s - mx)); z = ra(ex.sum()); a = ra(ex / z)
                o = ra(sum(a[j] * vals[j] for j in range(len(vals))))
                heads.append(dict(kv=g, keys=keys, vals=vals, dots=np.array(dots), s=s, max=mx, ex=ex, z=z, a=a, o=o))
            Lr["heads"] = heads
            concat = np.concatenate([hd["o"] for hd in heads])
            Lr["concat"] = concat
            ao = ra(W(p + "self_attn.o_proj.weight") @ concat)
            Lr["ao"] = ao
            x = ra(x + ao); Lr["x_mid"] = x
            Lr["ln2"] = rmsnorm(x, W(p + "post_attention_layernorm.weight"), ra)
            xh = Lr["ln2"]["y"]
            gt = ra(W(p + "mlp.gate_proj.weight") @ xh); up = ra(W(p + "mlp.up_proj.weight") @ xh)
            sig = ra(1 / (1 + np.exp(-gt))); act = ra(gt * sig * up)
            dn = ra(W(p + "mlp.down_proj.weight") @ act)
            Lr.update(gate=gt, up=up, sig=sig, act=act, down=dn)
            x = ra(x + dn); Lr["x_out"] = x
            P["layers"].append(Lr)
        P["x_final"] = x
        if pos >= len(ids) - 1 or pos >= 8:  # LM head only where the logits are used (last prompt token onward)
            P["lnf"] = rmsnorm(x, W("model.norm.weight"), ra)
            logits = ra(E @ P["lnf"]["y"])
            P["logits"] = logits
            nxt = int(np.argmax(logits))
            P["next"] = nxt
            srt = np.argsort(-logits)
            P["margin"] = float(logits[srt[0]] - logits[srt[1]])
            if pos == len(ids) - 1:
                out.append(nxt); ids.append(nxt)
        trace["positions"].append(P)
        pos += 1
        if len(out) == N_OUT:
            break
    trace["ids"] = ids
    return out, trace


def reference(model, ids):
    """transformers float32 logits at each decoding position for the given full id sequence."""
    import torch
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0].numpy().astype(np.float64)
    return lg


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tok, model, sd = load()
    ids = tok(PROMPT).input_ids
    n_in = len(ids)
    out, tr = forward(sd, ids, Rounder(None), Rounder(None))
    print("unrounded numpy:", out, [tok.decode([i]) for i in out])
    ref = reference(model, tr["ids"][:-1])
    for p in tr["positions"][n_in - 1:]:
        i = tr["positions"].index(p)
        print(f"  pos {i}: max |logit diff| vs transformers = {np.abs(p['logits'] - ref[i]).max():.2e}, margin {p['margin']:.4f}")
    for kw in (3, 4, 5):
        for ka in (3, 4, 5, 6):
            o, t = forward(sd, ids, Rounder(kw), Rounder(ka))
            ms = [p["margin"] for p in t["positions"] if "margin" in p]
            ok = o == out
            print(f"weights {kw} sig, activations {ka} sig: {'OK ' if ok else 'FAIL'} out={o} margins={[round(m, 3) for m in ms]}")
