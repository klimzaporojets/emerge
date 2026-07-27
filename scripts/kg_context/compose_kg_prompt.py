#!/usr/bin/env python3
"""COMPOSE_KG_PROMPT — turn AUGMENT_KG_CONTEXT component files into ready-to-run eval-sets.

TWO PROMPT MODES (selectable + comparable via `compose.prompt_modes`):
  - "unified"  : ONE prompt per instance, ALL ops' KG context + a unified instruction (like the existing
                 benchmark: extract everything in one pass; per-op analysis happens at SCORE).
                 Output: <output_dir>/unified/<condition_tag>/snapshot_<year>/delta_*.jsonl
  - "per_op"   : ONE prompt per (op, instance), with THAT op's gold (op-filtered) + a per-op instruction.
                 Output: <output_dir>/per_op/<op>/<condition_tag>/snapshot_<year>/delta_*.jsonl
Run both, then compare which prompting style helps each op (the KG-aware setup makes ops diverge more).

Everything is config-driven (nothing hardcoded): prompt_modes, operations, conditions, k (list -> sweep),
order, include_entities/relations/instruction, seed, unified_instruction, and per_operation overrides
(each op's conditions + instruction). `passage = passage + composed block`; heavy `kg_context` dropped;
a `kg_context_composed` stamp records mode/op/condition/instruction.

    PYTHONPATH=src python scripts/kg_context/compose_kg_prompt.py \
        --config config/kg_context/20260725_compose_ladder/config.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import sys

from kg_context import augment as A

try:
    from tqdm import tqdm
except Exception:
    def tqdm(*a, **k):
        return a[0] if a else None

OP_LABEL = {
    "x-triples": "exists", "e-triples": "add", "ee-triples": "mint_add",
    "ee-kg-triples": "infer", "d-triples": "deprecate",
}
ALL_OPS = list(OP_LABEL)
EXISTING_EDGE_OPS = {"x-triples", "d-triples"}
K_DEPENDENT = {"kg_rag", "oracle_pieces"}          # sweep `k` (candidate-pool depth) -> tag `<cond>_k<k>`
TRIPLE_N_SWEEP = {"kg_rag_triples", "both_triples"}  # sweep `max_triples` (kg-rag lookup depth) -> tag `<cond>_n<n>`
# REQUIRED compose keys — must be present (global compose or per_operation override) or HARD FAIL.
# ('instruction', 'existing_triples' are OPTIONAL feature toggles, handled separately.)
_REQUIRED_COMPOSE = ("conditions", "k", "order", "include_entities", "include_relations",
                     "include_instruction", "seed", "max_triples")
_UNIFIED_SRC = {"kg_rag": "kg_rag", "oracle_pieces": "oracle", "only_oracle": "oracle"}
# op_label -> template filename in `template_dir` (unified = op_label 'all').
TEMPLATE_FILE = {"all": "unified.txt", "exists": "exists.txt", "add": "add.txt",
                 "mint_add": "mint_add.txt", "infer": "infer.txt", "deprecate": "deprecate.txt"}


def load_templates(template_dir: str) -> tuple[dict, str]:
    """Load every op template + the shared output_format.txt from `template_dir` (HARD FAIL if missing)."""
    templates = {}
    for label, fn in TEMPLATE_FILE.items():
        with open(os.path.join(template_dir, fn)) as f:
            templates[label] = f.read()
    with open(os.path.join(template_dir, "output_format.txt")) as f:
        output_format = f.read().strip()
    return templates, output_format


# The operation enum in output_format.txt lists all five ops; a per-op prompt should ONLY offer its own op.
_OP_ENUM = "<EXISTS | ADD | MINT_ADD | INFER | DEPRECATE>"

def op_output_format(output_format: str, op_label: str) -> str:
    """Narrow the operation enum to the single op for a per-op prompt; unified ('all') keeps all five.
    HARD FAIL if the enum sentinel isn't present (catches a renamed output_format.txt instead of silently
    emitting the full 5-way enum)."""
    if op_label == "all":
        return output_format
    if _OP_ENUM not in output_format:
        raise ValueError(f"output_format.txt no longer contains the operation enum {_OP_ENUM!r}; "
                         f"cannot narrow it to op '{op_label}'. Fix op_output_format / output_format.txt.")
    return output_format.replace(_OP_ENUM, op_label.upper())   # exists->EXISTS, mint_add->MINT_ADD, ...


# few-shot pools: one file per op_label in `template_dir` (few_shot_<label>.txt), examples separated by this
# cut marker; built once (leakage-safe, 405B-filtered) by build_few_shot.py, consumed here.
FEWSHOT_SEP = "\n-----8<-----\n"


def load_few_shot(template_dir: str, n: int) -> dict | None:
    """Return {op_label: text of the FIRST n examples} for n>0, else None (zero-shot). HARD FAIL if a pool
    file is missing while n>0 (run build_few_shot.py --write first)."""
    if n <= 0:
        return None
    pools = {}
    for label in TEMPLATE_FILE:                       # exists, add, mint_add, infer, deprecate, all
        fp = os.path.join(template_dir, f"few_shot_{label}.txt")
        with open(fp) as f:
            exs = [e.strip() for e in f.read().split(FEWSHOT_SEP) if e.strip()]
        if len(exs) < n:
            raise ValueError(f"{fp}: only {len(exs)} few-shot examples but few_shot_n={n} requested")
        pools[label] = "\n\n".join(exs[:n])
    return pools


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _tag(condition: str, k: int) -> str:
    return f"{condition}_k{k}" if condition in K_DEPENDENT else condition


LLM_405B_V1 = "Meta-Llama-3.1-405B_prompt_v1"


def op_405b_supported(tkgu_triple: dict, op: str) -> bool:
    """405B Rule-C support verdict for ONE (gold triple, op). A TKGU op is GROUND TRUTH only if
    Meta-Llama-3.1-405B_prompt_v1 assessed it text-supported — THIS IS WHAT TABLE 3 SCORES, so we must
    neither predict nor oracle a 405B-false op. d-triples use the `triple_deprecation` verdict, every
    other op the `triple_assertion` verdict. Missing verdict -> NOT supported (excluded). Direct dict
    access on the assessment schema (hard-fail if malformed; no silent defaults)."""
    prompt_type = "triple_deprecation" if op == "d-triples" else "triple_assertion"
    for a in tkgu_triple["llm_assessment"]:
        if a["llm_name"] == LLM_405B_V1 and a["llm_prompt_type"] == prompt_type:
            return a["llm_assessment"] is True
    return False


def supported_op_keys(inst: dict) -> set[tuple[int, str]]:
    """{(gold_triple_idx, op)} for every (tkgu_triple, op) that is 405B-supported — the single source of
    truth for both the with-gold op set (instance_ops) and the oracle-content filter (driver)."""
    return {(idx, o) for idx, gt in enumerate(inst["tkgu_triples"])
            for o in (gt["tkgu_operations"] or []) if op_405b_supported(gt, o)}


def instance_ops(inst: dict) -> set[str]:
    """The ops for which this instance has a 405B-SUPPORTED gold (Rule-C). 405B-false ops are EXCLUDED,
    so per_op_all_instances=False never spends a prediction on a non-text-supported op."""
    return {o for (_, o) in supported_op_keys(inst)}


def _resolved_cfg(compose: dict, op: str | None) -> dict:
    """Effective compose settings: `compose` (global) keys with per_operation[op] overrides (per_op mode).
    REQUIRED keys must be present in global compose OR the per-op override -> KeyError (HARD FAIL) if not;
    no silent defaults. Optional feature toggles ('instruction', 'existing_triples') are None when absent.
    For unified (op is None) the instruction is `unified_instruction` (also optional)."""
    override: dict = {}
    if op is not None and "per_operation" in compose and op in compose["per_operation"]:
        override = compose["per_operation"][op]
    cfg: dict = {}
    for k in _REQUIRED_COMPOSE:
        if k in override:
            cfg[k] = override[k]
        elif k in compose:
            cfg[k] = compose[k]
        else:
            raise KeyError(f"compose config: required key '{k}' missing (global compose or per_operation[{op}])")
    for opt in ("instruction", "existing_triples"):   # optional: explicit None if not configured
        cfg[opt] = override[opt] if opt in override else (compose[opt] if opt in compose else None)
    if op is None:
        cfg["instruction"] = compose["unified_instruction"] if "unified_instruction" in compose else None
    return cfg


def op_compose_cfg(compose: dict, op: str) -> dict:
    return _resolved_cfg(compose, op)


def unified_compose_cfg(compose: dict) -> dict:
    return _resolved_cfg(compose, None)


def expand_points(cfg: dict) -> list[tuple[str, int, int, str]]:
    """Yield (condition, k, max_triples, tag). K_DEPENDENT conds sweep `k` (candidate-pool depth) -> `<cond>_k<k>`;
    TRIPLE_N_SWEEP conds (kg_rag_triples/both_triples) sweep `max_triples` (kg-rag lookup depth) -> `<cond>_n<n>`;
    all others are a single point at the default `max_triples`."""
    out, seen = [], set()
    dflt = int(cfg["max_triples"])
    for cond in cfg["conditions"]:
        if cond in K_DEPENDENT:
            for k in cfg["k"]:
                out.append((cond, int(k), dflt, _tag(cond, k)))
        elif cond in TRIPLE_N_SWEEP:      # reuse the `k` list as the max_triples sweep (per-op configurable)
            for n in cfg["k"]:
                out.append((cond, int(n), int(n), f"{cond}_n{n}"))
        else:
            tag = _tag(cond, 0)
            if tag not in seen:
                seen.add(tag)
                out.append((cond, int(cfg["k"][0]), dflt, tag))
    return out


def _emit(inst, kgc, rel, out_dir, mode, op, cond, k, mtri, tag, c, counts, templates, output_format, few_shot,
          allow_empty=False):
    """Render one instance's KG-aware prompt from the op's on-disk template and append it to its eval-set.
    `mtri` = this point's max_triples (swept for TRIPLE_N_SWEEP conds, else the default); `tag` = point tag.
    `allow_empty` (per_op_all_instances): emit oracle_triples even with NO gold for this op (empty KG context ->
    the negative/hallucination case: 'should return []') instead of skipping."""
    if cond == "oracle_triples":
        if op is not None and op not in EXISTING_EDGE_OPS:
            return
        if not allow_empty and not any((op is None or t["op"] == op) for t in kgc["oracle_triples"]):
            return
    # existing-triples CONTEXT on top of a PIECES condition (Add/Mint/Infer, unified). Source matches the
    # condition: kg_rag->kg_rag edges, oracle_pieces/only_oracle->oracle edges. UNIFIED always shows them;
    # PER_OP is gated by the op's `existing_triples` config {oracle:bool, kg_rag:bool}. `max_triples` caps
    # the kg_rag lookup edges everywhere (0 none / N first-N / -1 unlimited).
    inc_tri = None
    et = c["existing_triples"]   # None (optional), or {oracle:bool, kg_rag:bool} -> keys REQUIRED if present
    if mode == "unified":
        inc_tri = _UNIFIED_SRC[cond] if cond in _UNIFIED_SRC else None
    elif et is not None:  # per_op
        if cond in ("oracle_pieces", "only_oracle") and et["oracle"]:
            inc_tri = "oracle"
        elif cond == "kg_rag" and et["kg_rag"]:
            inc_tri = "kg_rag"
    op_label = OP_LABEL[op] if op is not None else "all"
    fields = A.template_fields(
        op_label, kgc, condition=cond, k=k, seed=c["seed"], order=c["order"],
        include_entities=c["include_entities"], include_relations=c["include_relations"],
        max_triples=mtri, op=op, include_triples=inc_tri)
    new = copy.deepcopy(inst)
    new.pop("kg_context", None)
    # Keep `passage` PRISTINE; the model-facing input goes in `prompt`, rendered from the op's template
    # (task wording + filled placeholders). PREDICT_TRIPLES feeds `prompt` verbatim (no baseline re-prompts);
    # `passage` stays canonical and its stored `hash_id` keeps the scoring join intact.
    new["passage"] = inst["passage"]
    fs = few_shot[op_label] if few_shot is not None else ""   # first-N examples for this op (or zero-shot)
    new["prompt"] = A.fill_template(templates[op_label], inst["passage"],
                                    fields, op_output_format(output_format, op_label), fs)
    new["kg_context_composed"] = {
        "prompt_mode": mode, "operation": op, "op_label": op_label,
        "condition": cond, "k": k, "max_triples": mtri, "order": c["order"],
        "include_entities": c["include_entities"], "include_relations": c["include_relations"],
        "template": TEMPLATE_FILE[op_label], "gold_retrieved": kgc["gold_retrieved"],
    }
    if mode == "per_op":
        sub, key = os.path.join("per_op", op_label, tag), f"per_op/{op_label}/{tag}"
    else:
        sub, key = os.path.join("unified", tag), f"unified/{tag}"
    dest = os.path.join(out_dir, sub, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "a") as g:
        g.write(json.dumps(new) + "\n")
    counts[key] = counts[key] + 1 if key in counts else 1


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    compose = cfg["compose"]
    comp_dir = cfg["component_dir"]
    out_dir = cfg["output_dir"]
    templates, output_format = load_templates(cfg["template_dir"])   # on-disk KG-aware prompt templates
    few_shot_n = compose["few_shot_n"]               # REQUIRED (hard fail); 0 = zero-shot, N = N examples/op
    few_shot = load_few_shot(cfg["template_dir"], few_shot_n)
    files = sorted(glob.glob(os.path.join(comp_dir, "snapshot_*", "delta_*.jsonl")))
    if not files:
        raise SystemExit(f"no component files under {comp_dir} (run AUGMENT_KG_CONTEXT first)")

    modes = compose["prompt_modes"]          # required (hard fail if absent) — no silent default
    operations = compose["operations"]       # required
    per_op_all = compose["per_op_all_instances"]  # required; True = emit EVERY op for EVERY instance (negative/
    #                                               hallucination case incl. ops the instance lacks); False = only ops it has
    for op in operations:
        if op not in OP_LABEL:
            raise ValueError(f"compose.operations has unknown op '{op}' (expected one of {list(OP_LABEL)})")
    op_cfg = {op: op_compose_cfg(compose, op) for op in operations}
    op_points = {op: expand_points(op_cfg[op]) for op in operations}
    uni_cfg = unified_compose_cfg(compose)
    uni_points = expand_points(uni_cfg)
    os.makedirs(out_dir, exist_ok=True)
    # Eval-set files are opened in APPEND mode below, so a re-run would accumulate stale/duplicate
    # prompts. Wipe THIS script's own generated subdirs (never out_dir itself) for a clean regen.
    for _sub in ("unified", "per_op"):
        _p = os.path.join(out_dir, _sub)
        if os.path.isdir(_p):
            shutil.rmtree(_p)

    print(f"COMPOSE_KG_PROMPT: {len(files)} component files | modes={modes} | "
          f"ops={[OP_LABEL[o] for o in operations]}", flush=True)
    if "unified" in modes:
        print(f"  unified <- {[t for *_, t in uni_points]}", flush=True)
    if "per_op" in modes:
        for op in operations:
            print(f"  per_op/{OP_LABEL[op]:9} <- {[t for *_, t in op_points[op]]}", flush=True)

    counts: dict[str, int] = {}
    total_inst = sum(1 for fp in files for ln in open(fp) if ln.strip())   # for the progress bar / ETA
    bar = tqdm(total=total_inst, desc="compose", unit="inst")
    for fp in files:
        rel = os.path.relpath(fp, comp_dir)
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                inst = json.loads(line)
                bar.update(1)
                bar.set_postfix(prompts=sum(counts.values()), refresh=False)
                kgc = inst["kg_context"]
                # 405B-SUPPORT FILTER (Rule-C, see op_405b_supported): keep ONLY oracle pieces belonging to a
                # 405B-text-supported gold (what Table 3 scores). oracle ENTITIES/RELATIONS are SHARED across
                # an instance's golds and augment dedups each to ONE entry keyed to its FIRST gold_triple_idx
                # (which may be a 405B-FALSE gold) -> filtering them by that stored idx WRONGLY drops a piece
                # that a *different* supported gold needs (e.g. "ASU" keyed to a false "employer" gold but
                # needed by the true "educated at" gold). So filter entities/relations by whether the
                # (qid/pid, op) is used by ANY 405B-supported gold; oracle_triples are per-gold full triples
                # (not shared) so keep the (gold_triple_idx, op) filter. kg_rag_* pools untouched.
                supported = supported_op_keys(inst)                    # {(gold_triple_idx, op)}
                sup_ent, sup_rel = set(), set()
                for _idx, _o in supported:
                    _gt = inst["tkgu_triples"][_idx]
                    _h, _p, _t = _gt["triple"]
                    sup_ent.add((_h, _o)); sup_ent.add((_t, _o))       # both ends (minted end's future qid harmless)
                    sup_rel.add((_p, _o))
                kgc["oracle_entities"]  = [x for x in kgc["oracle_entities"]  if (x["qid"], x["op"]) in sup_ent]
                kgc["oracle_relations"] = [x for x in kgc["oracle_relations"] if (x["pid"], x["op"]) in sup_rel]
                kgc["oracle_triples"]   = [x for x in kgc["oracle_triples"]   if (x["gold_triple_idx"], x["op"]) in supported]
                inst_ops = {o for (_, o) in supported} & set(operations)
                if not inst_ops:
                    continue
                if "unified" in modes:
                    for (cond, k, mtri, tag) in uni_points:
                        _emit(inst, kgc, rel, out_dir, "unified", None, cond, k, mtri, tag, uni_cfg, counts,
                              templates, output_format, few_shot)
                if "per_op" in modes:
                    for op in operations:                        # all configured ops (per_op_all) or only the
                        if not per_op_all and op not in inst_ops:  # instance's own ops
                            continue
                        for (cond, k, mtri, tag) in op_points[op]:
                            _emit(inst, kgc, rel, out_dir, "per_op", op, cond, k, mtri, tag, op_cfg[op], counts,
                                  templates, output_format, few_shot, allow_empty=per_op_all)
    bar.close()

    points = []
    if "unified" in modes:
        points += [{"mode": "unified", "operation": None, "condition": c, "k": k, "max_triples": mtri,
                    "tag": f"unified/{t}", "input_dir": os.path.join(out_dir, "unified", t)}
                   for (c, k, mtri, t) in uni_points]
    if "per_op" in modes:
        points += [{"mode": "per_op", "operation": op, "op_label": OP_LABEL[op], "condition": c, "k": k,
                    "max_triples": mtri, "tag": f"per_op/{OP_LABEL[op]}/{t}",
                    "input_dir": os.path.join(out_dir, "per_op", OP_LABEL[op], t)}
                   for op in operations for (c, k, mtri, t) in op_points[op]]
    manifest = {"component_dir": comp_dir, "template_dir": cfg["template_dir"], "prompt_modes": modes,
                "operations": operations, "compose": compose, "predict": cfg["predict"],
                "points": points, "counts": counts}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}", flush=True)
    print(f"manifest -> {os.path.join(out_dir, 'manifest.json')}", flush=True)
    print(f"=== COMPOSE_KG_PROMPT DONE. {len(counts)} eval-sets ({modes}) under {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
