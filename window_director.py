"""Geiru Window Director: build a Geiru Hybrid Windows render one window at a time.

Approve each window before the next one is made, re-roll the one you're looking at, or rewind to an earlier window
and redo it from there. When every window is made, Blend all runs the blended last steps over the whole video,
the part of Hybrid Windows that smooths the joins.

The node only puts its settings on the model. Geiru Hybrid Windows does the work, and runs exactly as
normal when Window Director isn't plugged in.

Each window is saved to output/geiru_window_director/<run_name>/ (only the latents that window made), so a run can be
picked up again after a restart and any earlier window can be rewound to.
"""
import json
import logging
import os
import shutil

import torch

CATEGORY = "Geiru Context Suite"
WHO = "Geiru Window Director"
DIR = "geiru_window_director"          # in ComfyUI's output folder
ACTIONS = ["reroll", "approve", "rewind", "blend", "replace"]
PREVIEWS = ["this window", "all windows so far"]
EVENT = "geiru_window_director.status"


# ---------------------------------------------------------------- run folder

def _base_dir():
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:
        base = "output"
    return os.path.join(base, DIR)


def _safe_name(name):
    name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name or "").strip())
    name = name.strip("._")
    return name or "run"


def _run_dir(name, make=True):
    d = os.path.join(_base_dir(), _safe_name(name))
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def _win_path(d, t):
    return os.path.join(d, f"window_{t:03d}.pt")


def _read_meta(d):
    p = os.path.join(d, "run.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[{WHO}] couldn't read {p}: {e}")
        return None


def _write_meta(d, meta):
    p = os.path.join(d, "run.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, p)


def _wipe(d):
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def _drop_from(d, t):
    """Delete the saved windows t, t+1, ... and their little previews."""
    for f in os.listdir(d):
        if f.startswith("window_") and f.endswith((".pt", THUMB_PNG, THUMB_WEBP)):
            try:
                if int(f[7:10]) >= t:
                    os.remove(os.path.join(d, f))
            except ValueError:
                pass


# ---------------------------------------------------------------- little previews for the strip

THUMB_PNG = "_thumb.png"        # the window's first frame, shown in its square
THUMB_WEBP = "_thumb.webp"      # the whole window, played when you hover over the square
THUMB_W = 256                   # width in pixels
_thumb_warned = []


def _thumb_base(d, t):
    return os.path.join(d, f"window_{t:03d}_thumb")


def _thumbs(d):
    """{window number: when its preview was saved}, so the page loads a new one after a re-roll."""
    out = {}
    try:
        for f in os.listdir(d):
            if f.startswith("window_") and f.endswith(THUMB_WEBP):
                try:
                    out[str(int(f[7:10]))] = int(os.path.getmtime(os.path.join(d, f)))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _save_thumb(d, t, lat, fps=24.0):
    """A rough preview of one window from its latents, the way ComfyUI's quick sampler preview makes one
    (latent2rgb): no VAE, on the CPU, well under a second. Never stops the render if it goes wrong."""
    try:
        from PIL import Image
        from comfy.latent_formats import MiniMaxH3Video as F
        if lat is None or lat.ndim != 5 or lat.shape[2] == 0:
            return
        fac = torch.tensor(F.latent_rgb_factors, dtype=torch.float32).t()          # 3 x 24
        bias = torch.tensor(F.latent_rgb_factors_bias, dtype=torch.float32)
        x = lat[0, :fac.shape[1]].detach().to("cpu", torch.float32).permute(1, 2, 3, 0)   # T, H, W, C
        rgb = torch.nn.functional.linear(x, fac, bias)                             # T, H, W, 3
        rgb = ((rgb + 1.0) / 2.0).clamp(0, 1).permute(0, 3, 1, 2)
        h = max(8, round(THUMB_W * rgb.shape[2] / max(1, rgb.shape[3])))
        rgb = torch.nn.functional.interpolate(rgb, size=(h, THUMB_W), mode="bilinear", align_corners=False)
        frames = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        imgs = [Image.fromarray(f) for f in frames]
        base = _thumb_base(d, t)
        imgs[0].save(base + ".tmp.png", format="PNG")
        os.replace(base + ".tmp.png", base + ".png")
        # one latent frame is 4 video frames (the first is 1), so play it at about the real speed
        imgs[0].save(base + ".tmp.webp", format="WEBP", save_all=True, append_images=imgs[1:],
                     duration=max(20, round(4000 / fps)), loop=0, quality=75)
        os.replace(base + ".tmp.webp", base + ".webp")
    except Exception as e:
        if not _thumb_warned:
            _thumb_warned.append(1)
            logging.warning(f"[{WHO}] couldn't make the little window preview (the render is fine): {e}")


def _aspect(meta):
    """Width / height of the video the take is made at, so every square in the strip has the video's shape."""
    try:
        v = (meta.get("sig") or {}).get("shapes")[0]
        return round(float(v[-1]) / float(v[-2]), 4) if v[-2] else None
    except Exception:
        return None


def _status(meta, name, msg=""):
    meta = meta or {}
    total = int(meta.get("total", 0) or 0)
    approved = int(meta.get("approved", 0) or 0)
    pending = int(meta.get("pending", 0) or 0)
    made = max(approved, pending)
    return {"run": _safe_name(name), "total": total, "approved": approved, "pending": pending,
            "all_made": bool(total) and made >= total, "blended": bool(meta.get("blended", False)),
            "joined": bool(meta.get("joined", False)), "msg": msg,
            "thumbs": _thumbs(_run_dir(name, make=False)), "aspect": _aspect(meta)}


def _send(status):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(EVENT, status)
    except Exception:
        pass


# ---------------------------------------------------------------- the directed render (called by Hybrid Windows)

def run_directed(d, api, state):
    """Render one window (or the blended finish) of a Hybrid Windows run, as Window Director asked.
    api: the pieces of Hybrid Windows' sampler pass. state: its st / pins / done lists (changed in place)."""
    windows, shapes, sigmas = api["windows"], api["shapes"], api["sigmas"]
    steps, k, h = api["steps"], api["k"], api["h"]
    dims, per_window, ix = api["dims"], api["per_window"], api["ix"]
    clean_parts, base_parts = api["clean_parts"], api["base_parts"]
    pack, unpack = api["pack"], api["unpack"]
    run_win, one_window, keep = api["run_win"], api["one_window"], api["keep"]
    counter, extra_args, ms = api["counter"], api["extra_args"], api["ms"]
    st, pins, done = state["st"], state["pins"], state["done"]
    work_dev, store, view_parts = api["dev"], api["store"], api["view_parts"]   # math on the GPU, copies in RAM
    n = len(windows)
    name = d.get("run_name", "run")
    folder = _run_dir(name)

    handoff_on = bool(getattr(h, "finished_handoff", False))
    # what must stay the same for the whole take (🎈 on the nodes). The keystone, keystone attention, capstone and
    # context noise settings can change between windows: they only change the windows made after the change.
    sig = {"shapes": [list(map(int, s)) for s in shapes], "windows": [list(map(int, w)) for w in windows],
           "sigmas": [round(float(x), 5) for x in sigmas], "handoff": handoff_on,
           "split": bool(getattr(h, "split_conds_to_windows", False)), "sound_first": bool(getattr(h, "sound_first", False))}
    meta = _read_meta(folder)
    if meta is not None and "k" in (meta.get("sig") or {}):          # a take saved before switch_step moved out
        meta["k"] = int(meta["sig"].pop("k"))
    if meta is None or not meta.get("sig"):
        meta = {"sig": sig, "k": int(k), "total": n, "approved": 0, "pending": 0, "rerolls": {}, "blended": False}
    else:
        old = meta["sig"]
        what = [label for key, label in (("shapes", "the length or size"), ("windows", "the window length or overlap"),
                                         ("sigmas", "the steps or scheduler"), ("handoff", "finished_handoff"),
                                         ("split", "split_conds_to_windows"), ("sound_first", "sound_first"))
                if key in old and old.get(key) != sig.get(key)]     # a take saved before a key existed: not checked
        if what:
            msg = (f"'{name}' was made with different settings ({', '.join(what)} changed). Set them back, "
                   f"or press Reset to start a new take")
            _send(_status(meta, name, msg))
            raise RuntimeError(f"{WHO}: {msg}.")
        meta["sig"] = {**old, **sig}
        for gone in ("keystone", "capstone"):                        # no longer fixed
            meta["sig"].pop(gone, None)
    meta["total"] = n
    k_take = int(meta.get("k", k))                 # the switch_step the windows were made at

    # ---- start state: fresh noise everywhere, nothing made yet
    start = unpack(ms.noise_scaling(sigmas[0], api["noise"], api["latent_image"]))
    for mi in range(len(shapes)):
        st[mi] = start[mi].to(device=st[mi].device, dtype=st[mi].dtype)
        pins[mi] = clean_parts[mi].clone()
        done[mi].clear()
    preview = [p.clone() for p in clean_parts]
    keys = {"key": None, "cap": None, "pass": 0}

    def load_upto(t):
        """Put windows 1..t back, from their saves."""
        for j in range(1, t + 1):
            p = _win_path(folder, j)
            if not os.path.isfile(p):
                raise RuntimeError(f"{WHO}: window {j} of '{name}' is missing from {folder}. Rewind to window {j} "
                                   f"to make it again.")
            w = torch.load(p, map_location="cpu", weights_only=True)
            for mi in range(len(shapes)):
                new = w["idx"][mi]
                if not new:
                    continue
                sl = ix(dims[mi], new)
                st[mi][sl] = w["st"][mi].to(device=st[mi].device, dtype=st[mi].dtype)
                pins[mi][sl] = w["pins"][mi].to(device=pins[mi].device, dtype=pins[mi].dtype)
                preview[mi][sl] = w["preview"][mi].to(device=preview[mi].device, dtype=preview[mi].dtype)
                done[mi].update(new)
            if not os.path.isfile(_thumb_base(folder, j) + ".webp"):     # a window saved before previews existed
                _save_thumb(folder, j, preview[0][ix(dims[0], per_window[j - 1][0])])
            if j == 1:
                dev, dt = st[0].device, st[0].dtype
                keys["key"] = None if w.get("key") is None else w["key"].to(device=dev, dtype=dt)
                keys["cap"] = None if w.get("cap") is None else w["cap"].to(device=dev, dtype=dt)
            keys["pass"] = int(w.get("pass", 0))

    def render(wi, reroll, merge=None):
        t = wi + 1
        idxs = windows[wi]
        new = [[i for i in per_window[wi][mi] if i not in done[mi]] for mi in range(len(shapes))]
        if reroll > 0:
            # a re-roll: fresh starting noise for the latents this window makes
            g = torch.Generator(device="cpu").manual_seed(
                (int(extra_args.get("seed", 0) or 0) + 1000003 * reroll + 7919 * t) % (2 ** 63))
            for mi in range(len(shapes)):
                if new[mi]:
                    sl = ix(dims[mi], new[mi])
                    fresh = torch.randn(st[mi][sl].shape, generator=g).to(device=work_dev, dtype=st[mi].dtype)
                    st[mi][sl] = ms.noise_scaling(sigmas[0], fresh, clean_parts[mi][sl].to(work_dev)).to(
                        device=store, dtype=st[mi].dtype)
        counter["pass"] = keys["pass"] + 9973 * reroll
        h.capstone, slot = None, None
        if wi > 0:
            if h.detail_swap and keys["key"] is not None and idxs[0] in done[0]:
                slot = idxs[0]
            if h.back_capstone and keys["cap"] is not None:
                h.capstone = keys["cap"]
        logging.info(f"[{WHO}] '{name}': window {t} of {n}" + (f" (re-roll {reroll})" if reroll else ""))
        kk = k_take
        out, x0 = run_win(0, kk, wi, st, slot, keys["key"])
        capture = None
        if kk < steps and (handoff_on or wi == 0):
            # finish this window on its own: with finished_handoff that's what the next window builds on (and what
            # you see); window 1 is always finished, for the keystone and capstone
            snap = [p.clone() for p in st]
            for mi in range(len(shapes)):
                sl = ix(dims[mi], per_window[wi][mi])
                snap[mi][sl] = out[mi][sl].to(snap[mi].dtype)
            if handoff_on and d.get("keep_late_steps", False):
                capture = []
                # each step: only the latents this window makes, copied straight to system RAM
                h.step_capture_fn = lambda x, _new=new: [
                    v[ix(dims[mi], [i - off for i in _new[mi]])].detach().to(store) if _new[mi] else torch.zeros(0)
                    for mi, (v, off) in enumerate(view_parts(x))]
            h.step_capture = capture
            try:
                if wi == 0:
                    fin, _ = one_window(sigmas[kk:], 0, snap)
                else:
                    fin, _ = run_win(kk, steps, wi, snap, slot, keys["key"])
            finally:
                h.step_capture = None
                h.step_capture_fn = None
            del snap
        elif kk < steps:
            fin = None
        else:
            fin = out
        # what you see: the finished window with finished_handoff on, else the step-switch_step guess the next
        # window builds on
        shown = fin if (handoff_on or kk >= steps) else (x0 if x0 is not None else out)
        if wi == 0:
            k0 = idxs[0]
            keys["key"] = api["detail"] if api["detail"] is not None else fin[0][:, :, k0:k0 + 1].clone()
            ci = idxs[5 * ((len(idxs) - 1) // 5)]
            keys["cap"] = fin[0][:, :, ci:ci + 1].clone()
        keep(wi, out, fin if (handoff_on and kk < steps) else x0)
        h.capstone = None
        save = {"idx": new, "st": [], "pins": [], "preview": [], "pass": int(counter["pass"])}
        settings = {"reroll": int(reroll), "keystone_swap": bool(h.detail_swap), "keystone_step": int(h.keystone_step),
                    "back_capstone": bool(h.back_capstone), "capstone_step": int(h.capstone_step),
                    "context_noise": float(h.context_noise), "context_noise_fade": str(h.context_noise_fade),
                    **(getattr(h, "ka_settings", None) or {})}
        meta.setdefault("window_settings", {})[str(t)] = settings
        for mi in range(len(shapes)):
            if new[mi]:
                sl = ix(dims[mi], new[mi])
                preview[mi][sl] = shown[mi][sl].to(preview[mi].dtype)
                save["st"].append(st[mi][sl].detach().cpu())
                save["pins"].append(pins[mi][sl].detach().cpu())
                save["preview"].append(preview[mi][sl].detach().cpu())
            else:
                save["st"].append(torch.zeros(0))
                save["pins"].append(torch.zeros(0))
                save["preview"].append(torch.zeros(0))
        if wi == 0:
            save["key"] = keys["key"].detach().cpu()
            save["cap"] = keys["cap"].detach().cpu()
        if capture:
            # the steps after switch_step (x at the start of each later step), so a later blend can start there
            late = {}
            for i, x in enumerate(capture):
                si = kk + i
                if si <= kk or si >= steps:
                    continue
                late[str(si)] = x                     # already just this window's latents, in system RAM
            save["late"] = late
        if merge is not None:
            # Replace: this pass may have re-made only part of the window. The file must still hold every frame the
            # window owns (its tail is what the next window is pinned to), so write those from the full state.
            own = merge["idx"]
            full = {"idx": own, "st": [], "pins": [], "preview": [], "pass": int(merge.get("pass", counter["pass"]))}
            for mi in range(len(shapes)):
                if own[mi]:
                    sl = ix(dims[mi], own[mi])
                    full["st"].append(st[mi][sl].detach().cpu())
                    full["pins"].append(pins[mi][sl].detach().cpu())
                    full["preview"].append(preview[mi][sl].detach().cpu())
                else:
                    full["st"].append(torch.zeros(0))
                    full["pins"].append(torch.zeros(0))
                    full["preview"].append(torch.zeros(0))
            for key in ("key", "cap"):
                if key in merge:
                    full[key] = merge[key]
            old_late, new_late = merge.get("late") or {}, save.get("late") or {}
            if old_late:
                late = {}
                for si, parts in old_late.items():
                    row = []
                    for mi in range(len(shapes)):
                        if not own[mi]:
                            row.append(torch.zeros(0))
                            continue
                        tmp = st[mi].detach().cpu().clone()
                        tmp[ix(dims[mi], own[mi])] = parts[mi].to(tmp.dtype)
                        if si in new_late and new[mi]:
                            tmp[ix(dims[mi], new[mi])] = new_late[si][mi].to(tmp.dtype)
                        row.append(tmp[ix(dims[mi], own[mi])].clone())
                    late[si] = row
                full["late"] = late
            save = full
        p = _win_path(folder, t)
        torch.save(save, p + ".tmp")
        os.replace(p + ".tmp", p)
        _save_thumb(folder, t, preview[0][ix(dims[0], per_window[wi][0])])     # the little preview in the strip
        del out, x0, fin, shown, capture
        api["free_vram"]()

    def made_so_far(wi=None):
        """Only part of the video, so the decode doesn't spend time on the blank rest of it: this window (with its
        overlap with the window before, to see the join) or every window made so far. The sampler's latent sizes
        are updated to match, so the latent comes out shorter."""
        if wi is not None and d.get("preview", PREVIEWS[0]) == PREVIEWS[0]:
            spans = [(min(per_window[wi][mi]), max(per_window[wi][mi]) + 1) if len(per_window[wi][mi]) else
                     (0, preview[mi].shape[dims[mi]]) for mi in range(len(shapes))]
        else:
            spans = [(0, (max(done[mi]) + 1) if done[mi] else preview[mi].shape[dims[mi]])
                     for mi in range(len(shapes))]
        parts = [preview[mi].narrow(dims[mi], a, b - a) for mi, (a, b) in enumerate(spans)]
        ls = getattr(api.get("model"), "latent_shapes", None)
        if len(parts) > 1:
            if not (isinstance(ls, list) and len(ls) == len(parts)):
                return pack(preview).to(work_dev)         # can't resize safely: whole length
            for i, p in enumerate(parts):
                ls[i] = p.shape
        return pack(parts).to(work_dev)

    action = d.get("action", "reroll")
    approved, pending = int(meta["approved"]), int(meta["pending"])
    rerolls = meta.setdefault("rerolls", {})

    if action == "blend":
        if meta.get("stale"):
            msg = (f"window(s) {meta['stale']} were built on frames that have since been replaced - replace them "
                   f"before Blend all")
            _send(_status(meta, name, msg))
            raise RuntimeError(f"{WHO}: {msg}.")
        if pending:
            approved, pending = max(approved, pending), 0
        if approved < n:
            msg = f"windows made: {approved} of {n} - make every window before Blend all"
            _send(_status(meta, name, msg))
            raise RuntimeError(f"{WHO}: {msg}.")
        load_upto(n)
        kb = int(k)                                      # switch_step now on Hybrid Windows
        if kb < k_take:
            msg = (f"'{name}' was made at switch_step {k_take}; it can only blend from that step or later. Set "
                   f"switch_step back, or press Reset to start a new take")
            _send(_status(meta, name, msg))
            raise RuntimeError(f"{WHO}: {msg}.")
        if k_take < kb < steps and not d.get("bypass_blend", False):
            # start the blend from the saved state of each window at step kb
            for j in range(1, n + 1):
                w = torch.load(_win_path(folder, j), map_location="cpu", weights_only=True)
                late = (w.get("late") or {}).get(str(kb))
                if late is None:
                    msg = (f"window {j} of '{name}' has no saved step {kb} (keep_late_steps was off, or "
                           f"finished_handoff was off). Set switch_step back to {k_take}")
                    _send(_status(meta, name, msg))
                    raise RuntimeError(f"{WHO}: {msg}.")
                for mi in range(len(shapes)):
                    new = w["idx"][mi]
                    if new:
                        st[mi][ix(dims[mi], new)] = late[mi].to(device=st[mi].device, dtype=st[mi].dtype)
        if kb >= steps and not d.get("bypass_blend", False):
            d = dict(d, bypass_blend=True)               # blend from the last step = nothing left to blend
        k = kb
        if d.get("bypass_blend", False):
            # no blend: join the windows' own finished versions, as they were approved
            if not api["handoff"]:
                logging.warning(f"[{WHO}] '{name}': bypass_blend with finished_handoff off - each window continued "
                                f"from the window before's unfinished guess, so the joins may not line up. Turn "
                                f"finished_handoff on in Hybrid Windows for joins without a blend.")
            logging.info(f"[{WHO}] '{name}': all {n} windows approved - joined as they are, blend steps skipped")
            meta.update(approved=n, pending=0, blended=True, joined=True)
            _write_meta(folder, meta)
            _send(_status(meta, name, "joined without blending"))
            return pack(preview).to(work_dev)
        logging.info(f"[{WHO}] '{name}': all {n} windows approved - blending steps {k + 1}-{steps} over the whole video")
        h.capstone = None
        if k >= steps:
            result = pack(st).to(work_dev) / max(1.0 - float(sigmas[-1]), 1e-8)
        else:
            li_p, m_p = pack(clean_parts).to(work_dev), pack(base_parts).to(work_dev)
            sk = float(sigmas[k])
            noise_w = (pack(st).to(work_dev) - (1.0 - sk) * li_p) / (sk * api["nscale"])
            api["free_vram"]()
            result = api["executor"](api["guider"], sigmas[k:], api["pass_args"](extra_args, keys["pass"] + 1),
                                     api["progress"](), noise_w, li_p, m_p, api["disable_pbar"])
        meta.update(approved=n, pending=0, blended=True, joined=False)
        _write_meta(folder, meta)
        _send(_status(meta, name, "blended"))
        return result

    if action == "replace":
        target = max(1, min(int(d.get("window", 1)), n))
        if pending:
            approved, pending = max(approved, pending), 0
        if approved < n:
            msg = f"windows made: {approved} of {n} - Replace needs every window made first (use Rewind instead)"
            _send(_status(meta, name, msg))
            raise RuntimeError(f"{WHO}: {msg}.")
        if target == 1:
            msg = "window 1 sets the keystone every later window was made with - it can't be replaced on its own"
            _send(_status(meta, name, msg))
            raise RuntimeError(f"{WHO}: {msg}.")
        load_upto(n)
        wi = target - 1
        old = torch.load(_win_path(folder, target), map_location="cpu", weights_only=True)
        free_tail = bool(d.get("free_tail", False)) and wi + 1 < n
        nxt = ([set(per_window[wi + 1][mi]) for mi in range(len(shapes))] if (wi + 1 < n and not free_tail)
               else [set() for _ in shapes])
        redo = [[i for i in old["idx"][mi] if i not in nxt[mi]] for mi in range(len(shapes))]
        if not redo[0]:
            raise RuntimeError(f"{WHO}: window {target} has no frames of its own to re-make.")
        rerolls[str(target)] = int(rerolls.get(str(target), 0)) + (1 if d.get("new_seed", True) else 0)
        reroll = rerolls[str(target)]
        fresh0 = unpack(ms.noise_scaling(sigmas[0], api["noise"], api["latent_image"]))
        for mi in range(len(shapes)):
            if redo[mi]:
                sl = ix(dims[mi], redo[mi])
                st[mi][sl] = fresh0[mi][sl].to(device=st[mi].device, dtype=st[mi].dtype)
                done[mi].difference_update(redo[mi])
        del fresh0
        prev = torch.load(_win_path(folder, target - 1), map_location="cpu", weights_only=True)
        keys["pass"] = int(prev.get("pass", 0))
        del prev
        count = int(meta.get("replaced", 0)) + 1
        backup = os.path.join(folder, f"window_{target:03d}.before-replace-{count}.bak")
        shutil.copyfile(_win_path(folder, target), backup)
        held = [len([i for i in per_window[wi][mi] if i in done[mi]]) for mi in range(len(shapes))]
        logging.info(f"[{WHO}] '{name}': replace window {target} of {n}: re-making {[len(r) for r in redo]} latents"
                     f"{' (hand-over included)' if free_tail else ''}, {held} held from its neighbours; the old window "
                     f"is kept as {os.path.basename(backup)}")
        render(wi, reroll, merge=old)
        stale = [t for t in meta.get("stale", []) if t != target]
        if free_tail and target + 1 not in stale:
            stale.append(target + 1)
        meta.update(approved=n, pending=0, blended=False, replaced=count, stale=sorted(stale))
        _write_meta(folder, meta)
        msg = (f"window {target} of {n} replaced" + (f"; window(s) {sorted(stale)} sit on old frames and must be replaced "
               f"next" if stale else " between its neighbours - look at it, then Blend all"))
        _send(_status(meta, name, msg))
        return made_so_far(wi)

    if action == "approve":
        if pending:
            approved, pending = pending, 0
        target = approved + 1
        reroll = int(rerolls.get(str(target), 0))
    elif action == "rewind":
        target = max(1, min(int(d.get("window", 1)), n))
        approved = min(approved, target - 1)
        rerolls[str(target)] = int(rerolls.get(str(target), 0)) + (1 if d.get("new_seed", True) else 0)
        reroll = rerolls[str(target)]
    else:                                                  # reroll (also what a plain Run does)
        target = pending or (approved + 1)
        if target > n:
            target, approved = n, n - 1
        if os.path.isfile(_win_path(folder, target)) or pending:
            rerolls[str(target)] = int(rerolls.get(str(target), 0)) + (1 if d.get("new_seed", True) else 0)
        reroll = int(rerolls.get(str(target), 0))

    if target > n:
        load_upto(n)
        meta.update(approved=n, pending=0)
        _write_meta(folder, meta)
        msg = f"all {n} windows approved - press Blend all"
        logging.info(f"[{WHO}] '{name}': {msg}")
        _send(_status(meta, name, msg))
        return made_so_far()

    _drop_from(folder, target)
    load_upto(target - 1)
    render(target - 1, reroll)
    meta.update(approved=target - 1, pending=target, blended=False)
    _write_meta(folder, meta)
    msg = f"window {target} of {n} ready to look at"
    _send(_status(meta, name, msg))
    return made_so_far(target - 1)


# ---------------------------------------------------------------- the node

class GeiruWindowDirector:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = ("Plug between Geiru Hybrid Windows and the sampler to build the render one window at "
                   "a time: Approve, Run/Re-roll, Rewind to selected, then Blend all once every window is made. "
                   "Use the node's buttons instead of ComfyUI's Run button (a plain Run re-rolls the current window).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "From Geiru Hybrid Windows."}),
                "run_name": ("STRING", {"default": "take1",
                                        "tooltip": "Each run name keeps its own windows, in "
                                                   "output/geiru_window_director/<run_name>. A new name starts a new "
                                                   "take without losing the old one."}),
                "rewind_to_window": ("INT", {"default": 1, "min": 1, "max": 9999,
                                             "tooltip": "The window Rewind to selected goes back to. Hidden: set by "
                                                        "clicking a square in the strip."}),
                "new_seed_on_reroll": ("BOOLEAN", {"default": False,
                                                   "tooltip": "On: Re-roll and Rewind give the window fresh noise, so "
                                                              "it comes out different each time. Off: the same noise, so "
                                                              "it only changes if you change its prompt or the seed."}),
                "keep_late_steps": ("BOOLEAN", {"default": False,
                                                "tooltip": "🎈 Keep it the same for the whole take: only windows made "
                                                           "while it's on have their later steps saved. Also save every step after switch_step for each window "
                                                           "(needs finished_handoff on). Then, once every window is "
                                                           "made, you can raise switch_step on Hybrid Windows "
                                                           "and Blend all starts from that later step, "
                                                           "without making the windows again. Uses more disk space."}),
                "bypass_blend": ("BOOLEAN", {"default": False,
                                             "tooltip": "On: the last button joins the windows as you approved them "
                                                        "(each with its own finish) and skips the blended steps. Needs "
                                                        "finished_handoff on in Hybrid Windows for clean "
                                                        "joins. Off: the blended steps over the whole video."}),
                "preview": (PREVIEWS, {"default": PREVIEWS[0],
                                       "tooltip": "What each window's render decodes. this window: only the new "
                                                  "window, with its overlap with the window before so you can see "
                                                  "the join (quickest). all windows so far: windows 1 to this one, "
                                                  "a running cut (takes longer as the video grows). Blend all "
                                                  "always gives the whole video."}),
                "action": (ACTIONS, {"default": "reroll", "tooltip": "Set by the buttons."}),
            },
            "optional": {
                "replace_free_tail": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Replace only: also re-make the frames this window hands "
                                                             "to the next one. The next window must then be replaced "
                                                             "too."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")      # every press of a button is a new render

    def patch(self, model, run_name="take1", rewind_to_window=1, new_seed_on_reroll=False, keep_late_steps=False,
              bypass_blend=False,
              preview=PREVIEWS[0],
              action="reroll", replace_free_tail=False):
        h = model.model_options.get("context_handler", None)
        if h is None or type(h).__name__ != "ClownContextHandler" or not hasattr(h, "switch_step"):
            raise RuntimeError(f"{WHO}: plug the model from Geiru Hybrid Windows into this node.")
        m = model.clone()
        m.model_options["geiru_window_director"] = {"run_name": run_name, "window": int(rewind_to_window),
                                                    "new_seed": bool(new_seed_on_reroll), "action": action,
                                                    "preview": preview if preview in PREVIEWS else PREVIEWS[0],
                                                    "bypass_blend": bool(bypass_blend),
                                                    "keep_late_steps": bool(keep_late_steps),
                                                    "free_tail": bool(replace_free_tail)}
        print(f"[{WHO}] '{run_name}': {action}" + (f" window {rewind_to_window}" if action in ("rewind", "replace") else ""))
        return (m,)


# ---------------------------------------------------------------- routes for the buttons

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/geiru_window_director/status")
    async def _route_status(request):
        name = request.query.get("run", "take1")
        d = _run_dir(name, make=False)
        return web.json_response(_status(_read_meta(d) if os.path.isdir(d) else None, name))

    @PromptServer.instance.routes.get("/geiru_window_director/thumb")
    async def _route_thumb(request):
        name = request.query.get("run", "take1")
        kind = request.query.get("kind", "png")
        try:
            t = int(request.query.get("t", "0"))
        except ValueError:
            t = 0
        if t < 1 or kind not in ("png", "webp"):
            return web.Response(status=400)
        p = _thumb_base(_run_dir(name, make=False), t) + "." + kind
        if not os.path.isfile(p):
            return web.Response(status=404)
        return web.FileResponse(p, headers={"Cache-Control": "no-cache"})

    @PromptServer.instance.routes.post("/geiru_window_director/reset")
    async def _route_reset(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = body.get("run", "take1")
        d = _run_dir(name, make=False)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        return web.json_response(_status(None, name, "run cleared - press Run/Re-roll to make window 1"))
except Exception as _e:
    logging.warning(f"[{WHO}] buttons' status routes not loaded: {_e}")


NODE_CLASS_MAPPINGS = {"GeiruWindowDirector": GeiruWindowDirector}
NODE_DISPLAY_NAME_MAPPINGS = {"GeiruWindowDirector": "Geiru Window Director"}
