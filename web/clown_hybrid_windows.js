// Geiru Hybrid Windows:
// - workflows saved with an older widget order get their values put back on the right widgets (by name).
//   now:    context_length, context_overlap, switch_step, fuse_method, freenoise, split_conds_to_windows,
//           crash_protection, then (advanced) keystone_swap, keystone_step, keystone_attention, balance, block_mode,
//           blocks, back_capstone, capstone_step, context_noise, context_noise_strength, context_noise_fade,
//           sound_first, finished_handoff
//   (take_from sits right after balance)
//   before take_from: the same without take_from
//   before keystone attention: the same without keystone_attention, balance, take_from, block_mode, blocks
//   prev:   ..., keystone_swap, back_capstone, finished_handoff, capstone_step, fuse_method, freenoise,
//           split_conds_to_windows, sound_first, context_noise, context_noise_strength, context_noise_fade,
//           crash_protection
//   old A/B: ..., back_capstone, capstone_step, fuse_method, freenoise, split_conds_to_windows, finished_handoff
//            (, sound_first)
// - settings that do nothing are greyed out: keystone_step and keystone_attention when keystone_swap is off,
//   balance / take_from / block_mode / blocks when keystone_attention is off, capstone_step when back_capstone is off,
//   context noise strength / fade when context_noise is off. block_mode = lighting blocks greys the blocks box and
//   shows the lighting blocks in it.
// - a balloon on the settings that must stay the same for a whole Window Director take, with a note under the title.
import { app } from "../../scripts/app.js";

const NO_KA = ["context_length", "context_overlap", "switch_step", "fuse_method", "freenoise", "split_conds_to_windows",
               "crash_protection", "keystone_swap", "keystone_step", "back_capstone", "capstone_step", "context_noise",
               "context_noise_strength", "context_noise_fade", "sound_first", "finished_handoff"];
const NO_TF = ["context_length", "context_overlap", "switch_step", "fuse_method", "freenoise", "split_conds_to_windows",
               "crash_protection", "keystone_swap", "keystone_step", "keystone_attention", "balance", "block_mode",
               "blocks", "back_capstone", "capstone_step", "context_noise", "context_noise_strength",
               "context_noise_fade", "sound_first", "finished_handoff"];
const PREV = ["context_length", "context_overlap", "switch_step", "keystone_swap", "back_capstone", "finished_handoff",
              "capstone_step", "fuse_method", "freenoise", "split_conds_to_windows", "sound_first", "context_noise",
              "context_noise_strength", "context_noise_fade", "crash_protection"];
const OLD = ["context_length", "context_overlap", "switch_step", "keystone_swap", "back_capstone", "capstone_step",
             "fuse_method", "freenoise", "split_conds_to_windows", "finished_handoff", "sound_first"];
const LIGHTING = "31-33, 36-44, 47";
// what a setting missing from an older workflow starts at
const DEFAULTS = { keystone_step: 0, keystone_attention: false, balance: 20, take_from: "context slice", block_mode: "lighting blocks", blocks: LIGHTING };
const BALLOON = "\u{1F388}";
const FIXED = ["context_length", "context_overlap", "switch_step", "split_conds_to_windows", "sound_first", "finished_handoff"];
const NOTE = BALLOON + " = Settings fixed between windows";
const NOTE_H = 18;
const SWITCHES = ["keystone_swap", "keystone_attention", "block_mode", "back_capstone", "context_noise"];

app.registerExtension({
    name: "clown.context.suite.hybrid_windows",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ClownHybridWindows") return;

        const get = (node, name) => node.widgets?.find((w) => w.name === name);
        // the height of the input sockets (settings turned into inputs don't count: they sit next to their setting)
        const slotsY = (node) => Math.max((node.inputs || []).filter((i) => !i.widget).length, node.outputs?.length || 0)
            * (globalThis.LiteGraph?.NODE_SLOT_HEIGHT ?? 20) + 6;

        const sync = (node) => {
            const on = (n) => get(node, n)?.value !== false;
            const dis = (n, d) => { const w = get(node, n); if (w) w.disabled = d; };
            const key = on("keystone_swap");
            dis("keystone_step", !key);
            dis("keystone_attention", !key);
            const ka = key && on("keystone_attention");
            dis("balance", !ka);
            dis("take_from", !ka);
            dis("block_mode", !ka);
            const mode = get(node, "block_mode");
            const box = get(node, "blocks");
            if (mode && box) {
                if (mode.value === "lighting blocks") {
                    if (box.value !== LIGHTING) node.__geiruCustomBlocks = box.value;
                    box.value = LIGHTING;
                } else if (node.__geiruLighting && node.__geiruCustomBlocks != null) {
                    box.value = node.__geiruCustomBlocks;
                }
                node.__geiruLighting = mode.value === "lighting blocks";
                box.disabled = !ka || mode.value === "lighting blocks";
            }
            dis("capstone_step", !on("back_capstone"));
            const cn = on("context_noise");
            dis("context_noise_strength", !cn);
            dis("context_noise_fade", !cn);
            node.setDirtyCanvas?.(true, true);
        };

        const mark = (node) => {
            for (const n of FIXED) { const w = get(node, n); if (w) w.label = `${BALLOON} ${n}`; }
            const dl = node.inputs?.find((i) => i.name === "detail_latent");
            if (dl) dl.label = `${BALLOON} detail_latent`;
            node.widgets_start_y = slotsY(node) + NOTE_H;
        };

        const hook = (node) => {
            for (const sw of SWITCHES) {
                const w = get(node, sw);
                if (!w || w.__geiruHooked) continue;
                const prevCb = w.callback;
                w.callback = function () { const out = prevCb?.apply(this, arguments); sync(node); return out; };
                w.__geiruHooked = true;
            }
        };

        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const r = computeSize ? computeSize.apply(this, arguments) : [this.size[0], this.size[1]];
            return [r[0], r[1] + NOTE_H];
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawForeground?.apply(this, arguments);
            if (!this.flags?.collapsed) {
                ctx.save();
                ctx.fillStyle = "#ffd54f";
                ctx.font = "12px sans-serif";
                ctx.fillText(NOTE, 10, slotsY(this) + 12);
                ctx.restore();
            }
            return r;
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            mark(this);
            hook(this);
            sync(this);
            const sz = this.computeSize();
            this.setSize?.([Math.max(this.size[0], sz[0]), Math.max(this.size[1], sz[1])]);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const v = info?.widgets_values;
            let layout = null, fill = {};
            if (Array.isArray(v) && v.length >= 10) {
                if (v.length === NO_TF.length && (v[11] === "lighting blocks" || v[11] === "custom blocks")) {
                    layout = NO_TF;                        // saved before take_from: block_mode in slot 11
                } else if (v.length === NO_KA.length && typeof v[3] === "string") {
                    layout = NO_KA;                        // saved before keystone attention: fuse_method in slot 3
                } else if (typeof v[3] === "boolean" && typeof v[6] === "number") {
                    layout = PREV;                         // keystone_swap in slot 3, capstone_step in slot 6
                } else if (typeof v[5] === "number") {
                    layout = OLD;                          // capstone_step in slot 5
                    if (v.length < 11) fill.sound_first = false;
                }
            }
            if (layout) {
                const saved = { ...fill };
                layout.forEach((name, i) => { if (i < v.length) saved[name] = v[i]; });
                for (const w of this.widgets || []) {
                    if (w.name in saved) w.value = saved[w.name];
                    else if (w.name in DEFAULTS) w.value = DEFAULTS[w.name];
                }
                this.__geiruCustomBlocks = null;
            }
            mark(this);
            hook(this);
            sync(this);
            return r;
        };
    },
});
