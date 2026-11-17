"""Geiru Keystone Attention.

Every token's attention is one pie that adds up to 100%: prompt, reference pictures, audio, the new frames,
the carried context (held frames from the last clip) and the keystone (clip 1's clean frame, grafted by
Geiru Keystone). This node leaves every slice exactly the size H3 would make it, except one: the slice
shared by the keystone and the carried context. Inside that slice only, `balance` moves attention from the
carried context to the keystone (balance > 1) or back (balance < 1). Prompt, references, audio and the new
frames keep exactly what they would normally get.

How it works (per attention call, per token, per head, exact):
- One normal attention pass, with two extra value channels that read out the keystone's share (pK) and the
  context's share (pC) of each token's attention.
- Two small passes over only the keystone keys and only the context keys give what each part contributes.
- Inside the shared slice S = pK + pC, the keystone gets S * b*pK / (b*pK + pC) instead of pK; the context
  gets the rest of S. The output is corrected by that shift. Everything outside S is untouched.

Needs Geiru Keystone in front (its av_latent goes into `latent`). Follows Geiru Context's
degradation_protection switch: off = pass through, like Keystone / Late Lock / Context Noise.
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

WHO = "Geiru Keystone Attention"

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


KEY = "geiru_keystone_balance"

# The blocks found to carry brightness on H3 (block search, Nov 2026): balance on these 13 holds a chain's brightness.
LIGHTING_BLOCKS = "31-33, 36-44, 47"
BLOCK_MODES = ["lighting blocks", "custom blocks"]
TAKE_FROM = ["context slice", "whole pie"]


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


class GeiruKeystoneAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model, after your other model patches (right before the guider)."}),
            "latent": ("LATENT", {"tooltip": "Geiru Keystone's av_latent output. Tells this node which latents are "
                                             "the keystone (and capstone) and whether degradation protection is on. "
                                             "Only read: the latent still goes from Keystone to the sampler as before."}),
            "balance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                "tooltip": "A weight on the keystone. 20 = each keystone token counts 20 times as much as it "
                           "normally would when attention is shared out. 1 = off.\n"
                           "context slice: shares out only between the keystone and the carried frames, so 20 ends up as "
                           "about 4x the keystone's normal attention (tops out around 6x).\n"
                           "whole pie: shares out with everything, so 20 ends up as about 8x (no limit).\n"
                           "Measured on the lighting blocks; varies by scene. The console shows the exact amount for "
                           "your shot when sampling starts."}),
            "block_mode": (BLOCK_MODES, {"default": "lighting blocks",
                "tooltip": "lighting blocks = the 13 blocks found to carry brightness (31-33, 36-44, 47). "
                           "custom blocks = type your own below."}),
            "blocks": ("STRING", {"default": LIGHTING_BLOCKS, "multiline": False,
                "tooltip": "Which transformer blocks (0-49) to apply it in, for custom blocks. Ranges and single "
                           "numbers, commas between: 31-33, 36-44, 47  or  31,32,33. Spaces don't matter."}),
            "enabled": ("BOOLEAN", {"default": True}),
            "take_from": (TAKE_FROM, {"default": "context slice",
                "tooltip": "Who gives up attention so the keystone can pull harder.\n"
                           "context slice (default, recommended): only the carried frames give it up; the prompt, "
                           "references, audio and new frames keep their exact share. Holds the keystone's look best, "
                           "at the cost of a slight sampling slowdown (about 5-10%).\n"
                           "whole pie: everything gives up a little. Close to no speed cost on any attention, but holds "
                           "the keystone's look less well."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Geiru Context Suite"
    DESCRIPTION = ("Moves attention between the keystone and the carried context only, keeping every other part of "
                   "H3's attention exactly the same. Plug Geiru Keystone's av_latent into latent. Follows the "
                   "degradation_protection switch on Geiru Context. Works with any attention setup (plain, Kijai's low "
                   "VRAM or Sage patches): on its own blocks it does the balanced attention itself, only where the balance "
                   "applies, and leaves everything else to whatever attention is set.")

    def patch(self, model, latent, balance, block_mode, blocks, enabled, take_from="context slice", **_old_settings):
        # **_old_settings: start_step / end_step from older workflows are ignored
        whole = str(take_from).strip().lower().startswith("whole")
        if not enabled or abs(balance - 1.0) < 1e-6:
            return (model,)
        if isinstance(latent, dict) and latent.get("geiru_protection", True) is False:
            print(f"[{WHO}] degradation protection off - passing through")
            return (model,)
        slots = sorted(set(int(s) for s in (latent.get("clown_keyframe_slots") or []))) if isinstance(latent, dict) else []
        has_back = isinstance(latent, dict) and latent.get("clown_keyframe_back") is not None
        if not slots and not has_back:
            print(f"[{WHO}] no keystone on this shot (first shot, Keystone off or not in front) - passing through")
            return (model,)

        chosen = parse_blocks(LIGHTING_BLOCKS if block_mode == "lighting blocks" else blocks)
        if not chosen:
            print(f"[{WHO}] no blocks given - passing through")
            return (model,)
        block_set = frozenset(chosen)
        print(f"[{WHO}] balance x{float(balance):g} on {len(chosen)} blocks ({block_mode}): {_ranges(chosen)}"
              f"{', taken from the whole pie' if whole else ''}")

        b = float(balance)
        stats = {"logged": False}
        m = model.clone()

        # 1) wrapper: per model call, find the keystone rows and the context rows (video rows only)
        def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
            state = {"key": None, "ctx": None, "cache": {}, "on": True}
            try:
                v = x[0] if isinstance(x, (list, tuple)) else x
                lt, lh, lw = v.shape[2], v.shape[3] + (v.shape[3] % 2), v.shape[4] + (v.shape[4] % 2)
                held = _held_video_latents(kwargs.get("denoise_mask"), lt)
                if held is not None:
                    held = F.pad(held.float(), (0, lw - held.shape[-1], 0, lh - held.shape[-2]), mode="replicate") > 0.5
                    held = held.reshape(lt, lh // 2, 2, lw // 2, 2).all(dim=4).all(dim=2)   # [lt, rows_h, rows_w]
                    anchor = torch.zeros(lt, dtype=torch.bool)
                    for s in slots:
                        if 0 <= s < lt:
                            anchor[s] = True
                    if has_back and held.shape[0] >= 1:
                        anchor[-1] = True
                    anchor &= held.reshape(lt, -1).all(dim=1).cpu()   # only fully held latents count as anchors
                    rows = held.reshape(lt, -1).cpu()
                    key = (anchor[:, None] & rows).reshape(-1)
                    ctx = (~anchor[:, None] & rows).reshape(-1)
                    if bool(key.any()) and bool(ctx.any()):
                        state["key"], state["ctx"] = key, ctx
            except Exception as exc:
                logging.warning("[%s] could not read the noise mask: %s", WHO, exc)
            transformer_options[KEY] = state
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
            print(f"[{WHO}] model doesn't look like MiniMax H3 - passing through")
            return (model,)
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
        return (m,)


NODE_CLASS_MAPPINGS = {"GeiruKeystoneAttention": GeiruKeystoneAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"GeiruKeystoneAttention": "Geiru Keystone Attention"}
