#!/usr/bin/env python3
"""Select + render KG-aware few-shot examples from the FULL 233K corpus (data/corpus/), with STRICT
leakage control against the 3,500-instance evaluation set and the Llama-405B (Rule-C) support filter.

Why: few-shot demos must NOT come from the eval set (data leakage) and must only use triples the text
actually supports (405B-v1 verdict True). We draw RICH instances (several verified triples of the same op)
from the corpus MINUS the eval hash_ids/passages, then render them in the KG-aware output format
(JSON with QIDs/PIDs; minted entities -> "MINTED").

  # PREVIEW candidates (ranked, per op) — review before writing:
  PYTHONPATH=src python scripts/kg_context/build_few_shot.py --preview --per-op 8
  # RENDER the chosen instances (by hash_id) into on-disk few-shot files under the template dir:
  PYTHONPATH=src python scripts/kg_context/build_few_shot.py --write \
      --pick add=HASH1,HASH2 deprecate=HASH3 ... unified=HASHX
"""
from __future__ import annotations
import argparse, glob, json, os, sys

LLM_405B_V1 = "Meta-Llama-3.1-405B_prompt_v1"
PROMPT_ASSERT, PROMPT_DEP = "triple_assertion", "triple_deprecation"
OP_LABEL = {"x-triples": "exists", "e-triples": "add", "ee-triples": "mint_add",
            "ee-kg-triples": "infer", "d-triples": "deprecate"}
OP_OUT = {"exists": "EXISTS", "add": "ADD", "mint_add": "MINT_ADD", "infer": "INFER", "deprecate": "DEPRECATE"}


def verdict(triple, prompt_type):
    for a in triple.get("llm_assessment", []):
        if a.get("llm_name") == LLM_405B_V1 and a.get("llm_prompt_type") == prompt_type:
            return a.get("llm_assessment") is True
    return False


def keep_triple_rule_c(triple):
    """405B Rule-C: op is text-supported (deprecation verdict for d-triples, else assertion verdict)."""
    ops = set(triple.get("tkgu_operations") or [])
    return verdict(triple, PROMPT_DEP) if "d-triples" in ops else verdict(triple, PROMPT_ASSERT)


def primary_op(triple):
    for o in (triple.get("tkgu_operations") or []):
        if o in OP_LABEL:
            return o
    return None


def load_eval_exclusion(eval_dir):
    hids, passages = set(), set()
    for fp in glob.glob(os.path.join(eval_dir, "**", "*.jsonl"), recursive=True):
        for l in open(fp):
            if l.strip():
                r = json.loads(l); hids.add(r["hash_id"]); passages.add(r["passage"].strip())
    return hids, passages


def iter_corpus(corpus_dir, excl_hids, excl_pass):
    """Yield corpus instances NOT in the eval set (hash_id AND passage-text checked)."""
    for fp in sorted(glob.glob(os.path.join(corpus_dir, "**", "*.jsonl"), recursive=True)):
        for l in open(fp):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["hash_id"] in excl_hids or r["passage"].strip() in excl_pass:
                continue
            yield r


def passes_qa(passage):
    """Mirror s07b_subsample_dataset.should_be_added_final_check — the SAME QA filter that produced the 3,500
    eval set: reject list-y / special-char-heavy / 'see also' stub passages, so few-shot picks match that
    realistic distribution (this is what dropped the 758-triple Olympics list-passage)."""
    import re
    if ":" not in passage:
        return True
    body = passage.split(":", 1)[1].strip()
    title = passage.split(":", 1)[0].strip()
    tot = body.count("|") + body.count(":")
    if tot > 0 and len(body.split(" ")) / tot < 3:            # too many separators -> list/table
        return False
    if len(re.findall(r"\*+", passage)) > 1:                  # multiple bullet groups
        frac = len(body.split(" ")) / len(re.findall(r"\*+", passage))
        if "list of" in title.lower() or "==cast==" in body.lower() or "== cast ==" in body.lower():
            return True
        if frac < 1:                                         # dense bullet list -> reject
            return False
    elif (body.lower().startswith("==see also==") or body.lower().startswith("== see also ==")) and len(body.split(" ")) < 2:
        return False
    return True


def verified_by_op(inst):
    """{op_label: [triple,...]} for triples that pass Rule-C, grouped by primary op."""
    out = {}
    for t in inst["tkgu_triples"]:
        op = primary_op(t)
        if op and keep_triple_rule_c(t):
            out.setdefault(OP_LABEL[op], []).append(t)
    return out


def _elem(qid, label, emerging):
    return {"label": label, "id": "MINTED"} if emerging else {"label": label, "id": qid}


def render_answer(triples):
    """The KG-aware JSON answer for a set of (already op-tagged) verified triples."""
    arr = []
    for t in triples:
        q, lab = t["triple"], t["triple_labels"]
        op = OP_LABEL[primary_op(t)]
        arr.append({"operation": OP_OUT[op],
                    "head": _elem(q[0], lab[0], t["emerging_head"]),
                    "relation": {"label": lab[1], "id": q[1]},
                    "tail": _elem(q[2], lab[2], t["emerging_tail"])})
    return json.dumps(arr, ensure_ascii=False)


FEWSHOT_SEP = "\n-----8<-----\n"   # must match compose_kg_prompt.load_few_shot
PICK_FILE = {"exists": "few_shot_exists.txt", "add": "few_shot_add.txt", "mint_add": "few_shot_mint_add.txt",
             "infer": "few_shot_infer.txt", "deprecate": "few_shot_deprecate.txt", "unified": "few_shot_all.txt"}


def _pairs(elems):
    seen, out = set(), []
    for q, l in elems:
        if q not in seen:
            seen.add(q); out.append(f"{l} ({q})")
    return out


def render_example(inst, op_label, tris):
    """One few-shot example in the KG-aware style: passage + the KG context it was given (relations/entities/
    existing triples with IDs) + the expected JSON answer. `op_label='unified'` -> all ops' verified triples."""
    rels = _pairs([(t["triple"][1], t["triple_labels"][1]) for t in tris])
    ents = _pairs([(t["triple"][i], t["triple_labels"][i]) for t in tris for i in (0, 2)
                   if not (t.get("emerging_head") if i == 0 else t.get("emerging_tail"))])
    lines = [f"Passage: {inst['passage']}",
             f"Candidate relation types: {', '.join(rels)}",
             f"Candidate existing KG entities: {', '.join(ents)}"]
    ops = {OP_LABEL[primary_op(t)] for t in tris}
    if ops & {"exists", "deprecate"}:   # for edge ops, the candidates ARE the existing triples
        et = [f"({t['triple_labels'][0]} ({t['triple'][0]}), {t['triple_labels'][1]} ({t['triple'][1]}), "
              f"{t['triple_labels'][2]} ({t['triple'][2]}))" for t in tris if OP_LABEL[primary_op(t)] in ("exists", "deprecate")]
        lines.append("Existing KG triples: " + "; ".join(et))
    lines.append("Answer: " + render_answer(tris))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--eval", default="data/evaluation_set")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--pick", nargs="*", default=[], help="op=HASH1,HASH2 ... (op in exists/add/mint_add/infer/deprecate/unified)")
    ap.add_argument("--out", default="src/kg_context/prompt_templates_kg_aware", help="template dir to write pools into")
    ap.add_argument("--per-op", type=int, default=8, help="how many ranked candidates to show per op")
    ap.add_argument("--min-triples", type=int, default=2, help="min verified triples of the op (richness)")
    ap.add_argument("--max-triples", type=int, default=6, help="max verified triples of the op (avoid giant list-passages)")
    ap.add_argument("--max-pass", type=int, default=700, help="max passage chars (prefer compact examples)")
    ap.add_argument("--max-unified", type=int, default=8, help="max total verified triples for a unified demo")
    ap.add_argument("--n", type=int, default=5, help="[--write auto] examples to keep per op (content-deduped, diverse)")
    args = ap.parse_args()

    excl_hids, excl_pass = load_eval_exclusion(args.eval)
    print(f"eval exclusion: {len(excl_hids)} hash_ids / {len(excl_pass)} passages", flush=True)

    if args.write and args.pick:   # EXPLICIT: render the picked hash_ids -> few_shot_<op>.txt pools
        picks = {}
        for tok in args.pick:
            op, hs = tok.split("=", 1)
            if op not in PICK_FILE:
                raise SystemExit(f"unknown op '{op}' (expected {list(PICK_FILE)})")
            picks.setdefault(op, []).extend(h for h in hs.split(",") if h)
        want = {h for hs in picks.values() for h in hs}
        found = {}
        for inst in iter_corpus(args.corpus, excl_hids, excl_pass):
            if inst["hash_id"] in want:
                found[inst["hash_id"]] = inst
                if len(found) == len(want):
                    break
        missing = want - set(found)
        if missing:
            raise SystemExit(f"picked hash_ids not found in NON-eval corpus (typo or leaked into eval?): {sorted(missing)}")
        os.makedirs(args.out, exist_ok=True)
        for op, hs in picks.items():
            exs = []
            for h in hs:
                vbo = verified_by_op(found[h])
                tris = [t for v in vbo.values() for t in v] if op == "unified" else vbo.get(op, [])
                if not tris:
                    raise SystemExit(f"{h}: no 405B-verified {'(any op)' if op=='unified' else op} triples")
                exs.append(render_example(found[h], op, tris))
            fp = os.path.join(args.out, PICK_FILE[op])
            with open(fp, "w") as f:
                f.write(FEWSHOT_SEP.join(exs) + "\n")
            print(f"wrote {len(exs)} example(s) -> {fp}", flush=True)
        return

    # collect per-op rich candidates + multi-op (unified) candidates
    by_op = {lbl: [] for lbl in OP_LABEL.values()}
    multi = []
    n = 0
    for inst in iter_corpus(args.corpus, excl_hids, excl_pass):
        n += 1
        vbo = verified_by_op(inst)
        if not vbo or len(inst["passage"]) > args.max_pass or not passes_qa(inst["passage"]):
            continue
        for lbl, tris in vbo.items():
            if args.min_triples <= len(tris) <= args.max_triples:   # rich but NOT a giant list-passage
                rels = {t["triple_labels"][1] for t in tris}
                by_op[lbl].append((len(rels), len(tris), inst["hash_id"],  # rank by relation diversity first
                                   inst["passage"], tris))
        total = sum(len(v) for v in vbo.values())
        if len(vbo) >= 3 and total <= args.max_unified:  # instance exercising many ops -> good unified demo
            multi.append((len(vbo), total, inst["hash_id"], inst["passage"], vbo))
    print(f"scanned {n} non-eval corpus instances", flush=True)

    def _dedup(sorted_cands, pass_idx):
        seen, out = set(), []
        for cand in sorted_cands:
            if cand[pass_idx] in seen:
                continue
            seen.add(cand[pass_idx]); out.append(cand)
        return out

    if args.write:   # AUTO: content-dedup (by entity set) + top-n diverse per op -> write pools
        def _sig(tris):
            return frozenset(q for t in tris for q in t["triple"] if str(q).startswith("Q"))

        def _pick(cands, n):
            seen, out = set(), []
            for c in sorted(cands, key=lambda x: (-x[0], -x[1])):
                s = _sig(c[4])
                if s in seen:
                    continue
                seen.add(s); out.append(c)
                if len(out) >= n:
                    break
            return out

        os.makedirs(args.out, exist_ok=True)
        for lbl in ["exists", "add", "mint_add", "infer", "deprecate"]:
            chosen = _pick(by_op[lbl], args.n)
            exs = [render_example({"hash_id": c[2], "passage": c[3]}, lbl, c[4]) for c in chosen]
            with open(os.path.join(args.out, PICK_FILE[lbl]), "w") as f:
                f.write(FEWSHOT_SEP.join(exs) + "\n")
            print(f"wrote {len(exs)} {lbl:9} example(s) -> {PICK_FILE[lbl]}  [{', '.join(c[2] for c in chosen)}]", flush=True)
        uni_cands = [(x[0], x[1], x[2], x[3], [t for v in x[4].values() for t in v]) for x in multi]
        uni = _pick(uni_cands, args.n)
        exs = [render_example({"hash_id": c[2], "passage": c[3]}, "unified", c[4]) for c in uni]
        with open(os.path.join(args.out, PICK_FILE["unified"]), "w") as f:
            f.write(FEWSHOT_SEP.join(exs) + "\n")
        print(f"wrote {len(exs)} unified   example(s) -> {PICK_FILE['unified']}  [{', '.join(c[2] for c in uni)}]", flush=True)
        return

    if args.preview:
        for lbl in ["exists", "add", "mint_add", "infer", "deprecate"]:
            cands = _dedup(sorted(by_op[lbl], key=lambda x: (-x[0], -x[1])), 3)[:args.per_op]
            print("\n" + "=" * 96 + f"\n### {lbl.upper()} — top {len(cands)} candidates ({args.min_triples}-{args.max_triples} verified {lbl} triples, diverse relations)\n" + "=" * 96)
            for nrel, ntri, hid, passage, tris in cands:
                print(f"\n[{hid}]  {ntri} verified triples / {nrel} distinct relations")
                print("  passage:", passage[:220].replace("\n", " "), "...")
                for t in tris:
                    q, lab = t["triple"], t["triple_labels"]
                    tag = []
                    if t.get("emerging_head"): tag.append("head=MINTED")
                    if t.get("emerging_tail"): tag.append("tail=MINTED")
                    print(f"     ({lab[0]} [{q[0]}], {lab[1]} [{q[1]}], {lab[2]} [{q[2]}])" + (f"  {tag}" if tag else ""))
        print("\n" + "=" * 96 + "\n### UNIFIED — top multi-op instances (>=3 distinct ops)\n" + "=" * 96)
        for nop, total, hid, passage, vbo in _dedup(sorted(multi, key=lambda x: (-x[0], -x[1])), 3)[:args.per_op]:
            print(f"\n[{hid}]  {nop} ops / {total} verified triples: {{ {', '.join(f'{k}:{len(v)}' for k,v in vbo.items())} }}")
            print("  passage:", passage[:220].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
