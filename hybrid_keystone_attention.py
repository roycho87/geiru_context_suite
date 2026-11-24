"""Keystone attention for Geiru Hybrid Windows (the same balanced attention as Geiru Keystone Attention).

Inside each window's own steps, the keystone (window 1's finished frame 0, locked into the window's first slot) and
the overlap carried from the windows before (the context) are both locked latents. On the chosen blocks, the slice of
attention shared by those two is re-split so the keystone gets `balance` times its normal pull and the context gives
up exactly what the keystone gains. Everything else (prompt, references, audio, new frames) is untouched.
Off on the blend steps (no keystone there) and on window 1 (no keystone yet).
"""
import logging
import math
import re
import types

import torch
import torch.nn.functional as F

import comfy.ldm.modules.attention as comfy_attention
import comfy.model_management
import comfy.patcher_extension
import comfy.quant_ops

WHO = "Geiru Hybrid Windows: keystone attention"
KEY = "geiru_hybrid_keystone_attention"


# Which ways of doing the balanced attention work on this machine: None = not tried yet, True / False once known.
_WORKS = {"sage": None, "own": None, "torch": None, "mask": None}


def _torch_lse(q, k, v, scale):
    """PyTorch's memory-efficient attention kernel, which also returns each row's log-sum-exp. q, k, v: [B, H, T, D]."""
    o, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False, scale=scale)[:2]
    return o, lse.float()[..., :q.shape[2]]


def _sage_lse(q, k, v, scale):
    """Sage attention that also returns each row's log-sum-exp (natural log, of the scaled scores)."""
    from sageattention import sageattn
    o, lse = sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=scale, return_lse=True)
    return o, lse.float().reshape(q.shape[0], q.shape[1], -1)[..., :q.shape[2]]


def _sage_plain(q, k, v):
    """Plain Sage attention, for cards where Sage can't return the log-sum-exp. q, k, v and the output: [B, H, T, D]."""
    from sageattention import sageattn
    return sageattn(q, k, v, tensor_layout="HND", is_causal=False)


def _under_sparse(override):
    """ComfyUI's Model Sparse Attention puts its own attention override on the hook. Returns (it is that one,
    the override that was there before it or None)."""
    if not getattr(override, "__module__", "").endswith("nodes_sparse_attention"):
        return False, None
    try:
        under = override.__closure__[override.__code__.co_freevars.index("previous")].cell_contents
    except Exception:
        under = None
    return True, under


def _try(name, what, fn):
    """Run one way of doing the balanced attention. The first time it fails it's switched off (and the next way is
    used); once it has worked, a later error is a real one and is raised."""
    if _WORKS[name] is False:
        return None
    try:
        r = fn()
    except Exception as exc:
        oom = isinstance(exc, getattr(comfy.model_management, "OOM_EXCEPTION", ())) or "out of memory" in str(exc).lower()
        if _WORKS[name] or oom:              # running out of memory says nothing about whether this way works
            raise
        _WORKS[name] = False
        logging.warning("[%s] %s doesn't work here (%s), using the next way (slower) until ComfyUI restarts",
                        WHO, what, exc)
        return None
    if _WORKS[name] is None:
        _WORKS[name] = True
        logging.info("[%s] balanced attention: %s", WHO, what)
    return r

class _Skip(Exception):
    pass


# The blocks found to carry brightness on H3 (block search, Nov 2026): balance on these 13 holds a chain's brightness.
LIGHTING_BLOCKS = "31-33, 36-44, 47"
BLOCK_MODES = ["lighting blocks", "custom blocks"]


def parse_blocks(text):
    """'31-33, 36-44, 47' or '31,32,33' or '31 - 33 ,47' -> sorted list of block numbers. Spaces don't matter."""
    text = re.sub(r"\s*-\s*", "-", str(text or ""))
    out = set()
    for part in re.split(r"[,;\s]+", text):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.update(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"[{WHO}] can't read '{part}' in blocks. Use numbers and ranges, like 31-33, 36-44, 47")
    return sorted(out)


def _ranges(blocks):
    out, i = [], 0
    while i < len(blocks):
        j = i
        while j + 1 < len(blocks) and blocks[j + 1] == blocks[j] + 1:
            j += 1
        out.append(str(blocks[i]) if i == j else f"{blocks[i]}-{blocks[j]}")
        i = j + 1
    return ", ".join(out)


def _held_video_latents(denoise_mask, latent_t):
    """bool [latent_t, H, W] (latent grid) - True where the video latent is held, or None."""
    if denoise_mask is None:
        return None
    m = denoise_mask[0, 0].to(torch.float32)
    if m.dim() != 3:
        return None
    if m.shape[0] == latent_t - 1:            # a back latent tacked on by Geiru Keystone's model patch: held
        m = torch.cat([m, torch.zeros_like(m[:1])], dim=0)
    if m.shape[0] != latent_t:
        return None
    return m < 0.5




def add_keystone_attention(m, balance, chosen, anchors_fn, whole=False):
    """Put keystone attention on model patcher m (already a clone), on blocks `chosen`.
    anchors_fn(lt): the window-local latents that are keystones on this model call (or None = off for this call);
    anchors_fn(None) is asked first, as a quick on/off check.
    whole: take the keystone's extra pull from the whole attention, not only from the overlap (the context slice)."""
    b = float(balance)
    stats = {"logged": False}

    # ComfyUI's Model Sparse Attention swaps a block's whole attention for its sparse one, which would skip this
    # node. On model calls where the balance applies, this node's blocks run with their normal attention instead;
    # every other block stays sparse.
    def keep_dense(topts):
        replace = topts.get("patches_replace")
        dit = replace.get("dit") if isinstance(replace, dict) else None
        if not dit:
            return
        new = None
        for i in chosen:
            fn = dit.get(("double_block", i))
            if fn is None or getattr(fn, "_geiru_keeps_dense", False):
                continue

            def run(args, extra, _fn=fn):
                original = extra["original_block"]

                def plain(a):
                    st = (a.get("transformer_options") or {}).get(KEY)
                    if a.get("attention") is not None and st and st.get("on", True) and st.get("key") is not None:
                        a = {k: v for k, v in a.items() if k != "attention"}
                        if not stats.get("sparse_note"):
                            stats["sparse_note"] = True
                            logging.info("[%s] sparse attention is on: this node's blocks (%s) stay on normal "
                                         "attention so the balance applies; the other blocks stay sparse",
                                         WHO, _ranges(chosen))
                    return original(a)
                return _fn(args, {**extra, "original_block": plain})
            run._geiru_keeps_dense = True
            if new is None:
                new = dict(dit)
            new[("double_block", i)] = run
        if new is not None:
            topts["patches_replace"] = {**replace, "dit": new}

    # 1) wrapper: per model call, find the keystone rows and the context rows (video rows only)
    def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
        state = {"key": None, "ctx": None, "cache": {}, "on": True}
        try:
            if anchors_fn(None) is None:              # nothing to balance on this call (blend steps, window 1, ...)
                raise _Skip()
            v = x[0] if isinstance(x, (list, tuple)) else x
            lt, lh, lw = v.shape[2], v.shape[3] + (v.shape[3] % 2), v.shape[4] + (v.shape[4] % 2)
            held = _held_video_latents(kwargs.get("denoise_mask"), lt)
            if held is not None:
                held = F.pad(held.float(), (0, lw - held.shape[-1], 0, lh - held.shape[-2]), mode="replicate") > 0.5
                held = held.reshape(lt, lh // 2, 2, lw // 2, 2).all(dim=4).all(dim=2)   # [lt, rows_h, rows_w]
                anchor = torch.zeros(lt, dtype=torch.bool)
                for s in (anchors_fn(lt) or []):
                    if -lt <= s < lt:
                        anchor[s] = True
                anchor &= held.reshape(lt, -1).all(dim=1).cpu()   # only fully held latents count as anchors
                rows = held.reshape(lt, -1).cpu()
                key = (anchor[:, None] & rows).reshape(-1)
                ctx = (~anchor[:, None] & rows).reshape(-1)
                if bool(key.any()) and bool(ctx.any()):
                    state["key"], state["ctx"] = key, ctx
        except _Skip:
            pass
        except Exception as exc:
            logging.warning("[%s] could not read the noise mask: %s", WHO, exc)
        transformer_options[KEY] = state
        if state["key"] is not None:
            try:
                keep_dense(transformer_options)
            except Exception as exc:
                logging.warning("[%s] could not keep its blocks off sparse attention: %s", WHO, exc)
        return executor(x, timestep, context, transformer_options, *args, **kwargs)

    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, wrapper)

    def indices(topts, seq_len, device):
        state = topts.get(KEY)
        layout = topts.get("minimax_h3_layout")
        if not state or not state.get("on", True) or state["key"] is None or layout is None or getattr(layout, "seq_len", None) != seq_len:
            return None
        ck = (id(layout), str(device))
        if ck in state["cache"]:
            return state["cache"][ck]
        res = None
        for a, z, kind in layout.segments:
            if kind == "video" and state["key"].numel() == z - a:
                ki = torch.nonzero(state["key"]).flatten() + a
                ci = torch.nonzero(state["ctx"]).flatten() + a
                res = (ki.to(device), ci.to(device))
                break
        state["cache"][ck] = res
        return res

    # 2) the balanced attention, for one head group: q, k, v are [1, h, T, D]. Whichever way is used, it needs
    #    the normal output, each token's keystone share and context share, and what the keystone and the context
    #    each contribute (two small passes over only those keys). The ways, fastest that fits first:
    #    - "sage": where Sage is this block's attention. One Sage pass that also returns each row's log-sum-exp;
    #      the shares are exp(log-sum-exp of that part - log-sum-exp of everything).
    #    - "own": any other attention the model is set to (PyTorch, Comfy Kitchen, ...). That attention runs
    #      twice: the normal pass, and once more over two marker channels, which reads the shares straight out.
    #    - "torch": PyTorch's memory-efficient kernel for everything (it returns the log-sum-exp too).
    #    - the old way: PyTorch attention with two extra value channels. Works everywhere, slowest.
    def balanced(qb, kb, vb, ki, ci, blk, route="torch", own=None):
        d = qb.shape[-1]
        scale = d ** -0.5
        B, H, T, _ = qb.shape
        S_len = kb.shape[2]
        if whole and own is not None and route == "own":
            # whole pie on an attention that takes a bias: the keystone's scores get +log(balance), one pass
            def with_bias():
                row = torch.zeros([1, S_len], dtype=kb.dtype, device=kb.device)
                row[0, ki] = math.log(max(b, 1e-6))
                return own(qb, kb, vb, mask=row)
            out = _try("mask", "a bias on the keystone, in the attention the model is set to", with_bias)
            if out is not None:
                if not stats["logged"]:
                    stats["logged"] = True
                    try:                                    # one extra measuring pass, once, for the console
                        _, lse = _torch_lse(qb, kb, vb, scale)
                        _, lseK = _torch_lse(qb, kb.index_select(2, ki), vb.index_select(2, ki), scale)
                        pK = torch.exp(lseK - lse)
                        log_shares(blk, ki, pK, b * pK / (1.0 + (b - 1.0) * pK))
                    except Exception:
                        logging.info("[%s] balance x%.2f from the whole pie on block %s", WHO, b, blk)
                return out.transpose(1, 2).reshape(B, T, H * d)
        kK, vK = kb.index_select(2, ki), vb.index_select(2, ki)
        kC, vC = (None, None) if whole else (kb.index_select(2, ci), vb.index_select(2, ci))

        def with_lse(main):
            out, lse = main()
            oK, lseK = _torch_lse(qb, kK, vK, scale)
            if whole:                                # whole pie: the context's part isn't needed
                return out, oK, None, torch.exp(lseK - lse).unsqueeze(-1), None
            oC, lseC = _torch_lse(qb, kC, vC, scale)
            return out, oK, oC, torch.exp(lseK - lse).unsqueeze(-1), torch.exp(lseC - lse).unsqueeze(-1)

        def with_own():
            out = own(qb, kb, vb)
            mark = vb.new_zeros(B, H, S_len, d)
            mark[:, :, ki, 0] = 1.0
            mark[:, :, ci, 1] = 1.0
            shares = own(qb, kb, mark)
            del mark
            pK = shares[..., 0:1].float().clamp_(0.0, 1.0)
            pC = shares[..., 1:2].float().clamp_(0.0, 1.0)
            del shares
            oK = F.scaled_dot_product_attention(qb, kK, vK, scale=scale)
            oC = None if whole else F.scaled_dot_product_attention(qb, kC, vC, scale=scale)
            return out, oK, oC, pK, pC

        r = None
        if route == "sage":
            r = _try("sage", "Sage", lambda: with_lse(lambda: _sage_lse(qb, kb, vb, scale)))
        if r is None and own is not None and route in ("sage", "own"):
            r = _try("own", "the attention the model is set to", with_own)
        if r is None:
            r = _try("torch", "PyTorch's memory-efficient kernel", lambda: with_lse(lambda: _torch_lse(qb, kb, vb, scale)))
        if r is not None:
            out, oK, oC, pK, pC = r
            return finish(out, oK, oC, pK, pC, ki, ci, blk, B, T, H, d)
        # the old way: full pass, two extra value channels = keystone share and context share
        ind = vb.new_zeros(S_len, 2)
        ind[ki, 0] = 1.0
        ind[ci, 1] = 1.0
        pad = (-(d + 2)) % 8
        v2 = torch.cat([vb, ind.view(1, 1, S_len, 2).expand(B, H, S_len, 2), vb.new_zeros(B, H, S_len, pad)], dim=-1)
        full = F.scaled_dot_product_attention(qb, kb, v2, scale=scale)
        out = full[..., :d]
        pK = full[..., d:d + 1].float()
        pC = full[..., d + 1:d + 2].float()
        del full, v2
        oK = F.scaled_dot_product_attention(qb, kK, vK, scale=scale)
        oC = None if whole else F.scaled_dot_product_attention(qb, kC, vC, scale=scale)
        return finish(out, oK, oC, pK, pC, ki, ci, blk, B, T, H, d)

    def log_shares(blk, ki, pK, new_pK, pC=None, new_pC=None):
        a, n = float(pK.float().mean()), float(new_pK.float().mean())
        msg = (f"[{WHO}] balance x{b:g} ({'whole pie' if whole else 'context slice'}), block {blk}, first step: "
               f"keystone gets {a * 100:.1f}% -> {n * 100:.1f}% of the attention ({n / max(a, 1e-9):.1f}x)")
        if pC is not None:
            msg += f", carried frames {float(pC.mean()) * 100:.1f}% -> {float(new_pC.mean()) * 100:.1f}%"
        logging.info(msg)

    def finish(out, oK, oC, pK, pC, ki, ci, blk, B, T, H, d):
        if whole:
            # the keystone's weights x balance, then everything renormalised: every other part gives up a little
            z = 1.0 + (b - 1.0) * pK
            out = ((out.float() + (b - 1.0) * pK * oK.float()) / z).to(out.dtype)
            if not stats["logged"]:
                stats["logged"] = True
                log_shares(blk, ki, pK, b * pK / z)
            return out.transpose(1, 2).reshape(B, T, H * d)
        sh = pK + pC
        denom = b * pK + pC
        new_pK = torch.where(denom > 1e-12, sh * b * pK / denom.clamp_min(1e-12), pK)
        delta = new_pK - pK                                         # keystone gains delta, context loses delta
        out = (out.float() + delta * (oK.float() - oC.float())).to(out.dtype)
        if not stats["logged"]:
            stats["logged"] = True
            log_shares(blk, ki, pK, new_pK, pC, sh - new_pK)
        return out.transpose(1, 2).reshape(B, T, H * d)

    # 3) this node's own attention forward on its blocks. It replaces whatever attention forward is there (H3's,
    #    Kijai's low VRAM or Sage patches, ...), but only acts on calls where the balance applies; every other call
    #    goes to that previous forward untouched. Takes the input as a tensor or as Kijai's one-item list.
    def make_forward(prev, blk, stock):
        prev_fn = getattr(prev, "__func__", prev)
        prev_sage = "sage" in getattr(prev_fn, "__name__", "").lower()                 # Kijai's Sage patch
        through_comfy = stock or bool(getattr(prev_fn, "_uses_optimized_attention", False))  # H3's own / low VRAM

        def how(attn, topts):
            """Which attention this block would use without this node -> (route, own attention or None)."""
            if prev_sage:
                # a Sage patch (Kijai's, the AMD one): Sage with the log-sum-exp when that works on this card,
                # else plain Sage twice
                return "sage", _sage_plain
            if not through_comfy:
                return "torch", None                 # some other attention patch: can't call it, so PyTorch
            override = topts.get("optimized_attention_override")
            if override is not None:
                sparse, under = _under_sparse(override)
                if sparse:                           # this block stays dense: call what is under the sparse override
                    topts = dict(topts)
                    if under is None:
                        topts.pop("optimized_attention_override", None)
                    else:
                        topts["optimized_attention_override"] = under
            pref = getattr(attn, "comfy_attention", None) if stock else None
            kw = {"preferred_attention": pref} if pref is not None else {}

            def own(qg, kg, vg, mask=None):
                h = qg.shape[1]
                o = comfy_attention.optimized_attention(qg, kg, vg, h, mask=mask, skip_reshape=True,
                                                        transformer_options=topts, **kw)
                return o.reshape(qg.shape[0], qg.shape[2], h, -1).transpose(1, 2)

            picked = "optimized_attention_override" in topts or getattr(pref, "function", None) is not None
            default = getattr(comfy_attention.optimized_attention, "__name__", "")
            return ("sage" if not picked and default == "attention_sage" else "own"), own

        def forward(self, x, rope_freqs=None, transformer_options={}):
            topts = transformer_options if isinstance(transformer_options, dict) else {}
            state = topts.get(KEY)
            if not state or not state.get("on", True) or state.get("key") is None:
                return prev(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
            xt = x[0] if isinstance(x, list) else x
            s_len = xt.shape[0]
            idx = indices(topts, s_len, xt.device)
            if idx is None:
                return prev(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
            if isinstance(x, list):
                x = x.pop()                                          # Kijai's low VRAM block hands h over in a list
            ki, ci = idx
            device, dtype = x.device, x.dtype
            q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
            del x
            v = v.view(s_len, self.heads, self.head_dim)
            if rope_freqs is not None:
                q = q.view(1, s_len, self.heads, self.head_dim)
                k = k.view(1, s_len, self.heads, self.head_dim)
                qw = comfy.model_management.cast_to(self.q_norm.weight, device=device)
                kw = comfy.model_management.cast_to(self.k_norm.weight, device=device)
                rot = rope_freqs.shape[-3] * 2
                if comfy.model_management.in_training:
                    q, k = comfy.quant_ops.ck.rms_rope_split_half(
                        q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                else:
                    comfy.quant_ops.ck.rms_rope_split_half_(
                        q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                q = q[0]
                k = k[0]
            else:
                q = self.q_norm(q.view(s_len, self.heads, self.head_dim))
                k = self.k_norm(k.view(s_len, self.heads, self.head_dim))
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            # head groups keep the memory down (heads are independent, so the result is the same);
            # follows Kijai's low VRAM head_chunks when it's set
            n = max(1, min(int(topts.get("minimax_head_chunks", 4) or 4), self.heads))
            route, own = how(self, topts)
            out = torch.empty((s_len, self.heads * self.head_dim), dtype=dtype, device=device)
            hs = 0
            for size in [self.heads // n + (1 if i < self.heads % n else 0) for i in range(n)]:
                he = hs + size
                o = balanced(q[:, hs:he], k[:, hs:he], v[:, hs:he], ki, ci, blk, route, own)
                out[:, hs * self.head_dim:he * self.head_dim] = o.squeeze(0)
                hs = he
            del q, k, v
            return self.out_proj(out)
        return forward

    diffusion_model = m.get_model_object("diffusion_model")
    h3_blocks = getattr(diffusion_model, "blocks", None)
    if not h3_blocks or not hasattr(h3_blocks[0], "attn") or not hasattr(h3_blocks[0].attn, "qkv_proj"):
        print(f"[{WHO}] model doesn't look like MiniMax H3 - keystone attention not added")
        return m
    others = set()
    for i in chosen:
        if i >= len(h3_blocks):
            continue
        attn = h3_blocks[i].attn
        key_name = f"diffusion_model.blocks.{i}.attn.forward"
        prev = m.object_patches.get(key_name)
        stock = prev is None
        if not stock:
            others.add(getattr(getattr(prev, "__func__", prev), "__name__", "patch"))
        else:
            prev = types.MethodType(type(attn).forward, attn)
        m.add_object_patch(key_name, types.MethodType(make_forward(prev, i, stock), attn))
    if others:
        print(f"[{WHO}] sits on top of the attention patch already on those blocks ({', '.join(sorted(others))}); "
              f"it takes over only where the balance applies")
    return m


