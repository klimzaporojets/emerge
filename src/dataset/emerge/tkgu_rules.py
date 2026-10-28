"""The TKGU-operation rules against the KG G_t = (V_t, R_t, T_t) at the snapshot date t (manuscript §3).

  op                         triple       entity to mint (flagged emerging)   other / existing entities
  Exists     (x-triples)     in T_t       none                                 in KG
  Add        (e-triples)     not in T_t   none                                 both in KG
  Mint+Add   (ee-triples)    not in T_t   >= 1, each NOT in KG                 non-minted one in KG
  Infer      (ee-kg-triples) not in T_t   exactly 1, NOT in KG                 the other one in KG
  Deprecate  (d-triples)     NO rule: a text may deprecate a fact that is or is not in T_t, also one of an entity
                             being minted (the paper's own example) -> never checked, never dropped

"Entity in KG" = the entity appears in >= 1 triple of the snapshot OR in the snapshot's entity dictionary (the
entity universe KG-aware models link against). An operation that breaks a rule is dropped; one exception: an Add
whose triple is already in T_t (both entities existing and in the KG) is an Exists and is relabelled as such,
unless the instance already has that Exists (then it is only dropped, no duplicate).

Used by the construction step `s06_refactor_final_format.py` and by the final check
`scripts/kg_context/check_snapshot_consistency.py`, so construction and check apply the same rules. The KG files are
STREAMED once per snapshot date (never decompressed); only the wanted triples / entities are kept.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

from kg_context.snapshot_io import open_kg_snapshot

logger = logging.getLogger(__name__)

EXISTS, ADD, MINT_ADD, INFER, DEPRECATE = 'x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples', 'd-triples'
OPS = (EXISTS, ADD, MINT_ADD, INFER, DEPRECATE)
OP_LABEL = {EXISTS: 'Exists', ADD: 'Add', MINT_ADD: 'Mint+Add', INFER: 'Infer', DEPRECATE: 'Deprecate'}
# triple-rule reasons
NOT_IN_T, IN_T = 'not_in_T', 'in_T'
# entity-rule reasons
FLAG_STRUCTURE, MINTED_IN_KG, EXISTING_NOT_IN_KG = 'flag_structure', 'minted_in_KG', 'existing_not_in_KG'
# what happens to an operation that breaks a rule
REMOVED, RELABELLED_TO_EXISTS, REMOVED_DUPLICATE_OF_EXISTS = 'removed', 'relabelled_to_exists', \
    'removed_duplicate_of_exists'
PROGRESS_SECS = 30

Triple = Tuple[str, str, str]


def _check_op(op: str) -> None:
    if op not in OPS:
        raise ValueError(f'unknown TKGU operation {op!r} (known: {OPS})')


def triple_violation(op: str, in_t: bool) -> Optional[str]:
    """Triple rule: Exists needs (s,p,o) IN T_t; Add / Mint+Add / Infer need it NOT in T_t; Deprecate has no rule."""
    _check_op(op)
    if op == DEPRECATE:
        return None
    if op == EXISTS:
        return None if in_t else NOT_IN_T
    return IN_T if in_t else None


def flags_violate(op: str, emerging_head: bool, emerging_tail: bool) -> bool:
    """The emerging flags must fit the op: Mint+Add >= 1 emerging, Infer exactly 1, Exists / Add none."""
    _check_op(op)
    if op == DEPRECATE:
        return False
    if op == MINT_ADD:
        return not (emerging_head or emerging_tail)
    if op == INFER:
        return emerging_head == emerging_tail
    return emerging_head or emerging_tail


def entity_violation(op: str, emerging_head: bool, emerging_tail: bool,
                     head_in_kg: bool, tail_in_kg: bool) -> Optional[str]:
    """Entity rule: flags fit the op; an entity to mint must NOT be in the KG, an existing one must be in the KG."""
    _check_op(op)
    if op == DEPRECATE:
        return None
    if flags_violate(op, emerging_head, emerging_tail):
        return FLAG_STRUCTURE
    for emerging, in_kg in ((emerging_head, head_in_kg), (emerging_tail, tail_in_kg)):
        if emerging and in_kg:
            return MINTED_IN_KG
        if not emerging and not in_kg:
            return EXISTING_NOT_IN_KG
    return None


def op_violation(op: str, in_t: bool, emerging_head: bool, emerging_tail: bool,
                 head_in_kg: bool, tail_in_kg: bool) -> Tuple[Optional[str], Optional[str]]:
    """(triple-rule reason, entity-rule reason); the operation is consistent iff both are None."""
    return (triple_violation(op, in_t),
            entity_violation(op, emerging_head, emerging_tail, head_in_kg, tail_in_kg))


def decide_ops(ops: List[str], key: Triple, emerging_head: bool, emerging_tail: bool, in_t: bool,
               head_in_kg: bool, tail_in_kg: bool, exists_in_instance: Set[Triple]):
    """Apply the rules to the operations of one triple entry.

    Returns (kept_ops, decisions): kept_ops in their original order (a relabelled Add becomes 'x-triples' in its
    place); decisions = [(op, action, triple_reason, entity_reason)] for every operation that broke a rule.
    `exists_in_instance` = the (s,p,o) that already have an Exists in this instance; a relabelled Add is added to it.
    """
    kept, decisions = [], []
    for op in ops:
        tr, er = op_violation(op, in_t, emerging_head, emerging_tail, head_in_kg, tail_in_kg)
        if tr is None and er is None:
            kept.append(op)
            continue
        action = REMOVED
        if op == ADD and tr == IN_T and er is None:
            if key in exists_in_instance:
                action = REMOVED_DUPLICATE_OF_EXISTS
            else:
                action = RELABELLED_TO_EXISTS
                kept.append(EXISTS)
                exists_in_instance.add(key)
        decisions.append((op, action, tr, er))
    return kept, decisions


def triple_key(triple: Iterable) -> Triple:
    s, p, o = triple
    return str(s), str(p), str(o)


# ---------------------------------------------- KG lookups ----------------------------------------------

class KgLookup:
    """What the KG at one snapshot date says about the wanted triples / entities."""

    def __init__(self, triples_in_t: Set[Triple], entities_in_triples: Set[str], entities_in_dictionary: Set[str]):
        self.triples_in_t = triples_in_t
        self.entities_in_triples = entities_in_triples
        self.entities_in_dictionary = entities_in_dictionary

    def in_t(self, key: Triple) -> bool:
        return key in self.triples_in_t

    def in_kg(self, entity: str) -> bool:
        return entity in self.entities_in_triples or entity in self.entities_in_dictionary


def snapshot_file(kg_snapshots_dir: str, snapshot_date: str) -> str:
    return os.path.join(kg_snapshots_dir, f'kg_snapshot_{snapshot_date}.tsv.gz')


def entity_dictionary_file(entity_dicts_dir: str, snapshot_date: str) -> str:
    """`<dir>/<date>/documents.jsonl.gz` (the released ReLiK entity index layout), or the plain `.jsonl`."""
    path = os.path.join(entity_dicts_dir, snapshot_date, 'documents.jsonl.gz')
    if os.path.exists(path):
        return path
    if os.path.exists(path[:-3]):
        return path[:-3]
    raise FileNotFoundError(f'entity dictionary not found: {path} (nor {path[:-3]})')


def scan_snapshot(path: str, want_triples: Set[Triple], want_entities: Set[str]) -> Tuple[Set[Triple], Set[str]]:
    """Stream one snapshot once: (wanted triples in T_t, wanted entities that appear in >= 1 triple)."""
    found_triples, found_entities = set(), set()
    start = last = time.time()
    nr = 0
    with open_kg_snapshot(path) as f:                      # raises if the snapshot is missing
        for nr, line in enumerate(f, start=1):
            parts = line.rstrip('\n').rstrip('\r').split('\t')
            if len(parts) < 3:
                raise ValueError(f'{path}:{nr}: expected >= 3 TAB-separated fields, got {len(parts)}')
            s, p, o = parts[0], parts[1], parts[2]
            if s in want_entities:
                found_entities.add(s)
            if o in want_entities:
                found_entities.add(o)
            if (s, p, o) in want_triples:
                found_triples.add((s, p, o))
            if time.time() - last >= PROGRESS_SECS:
                logger.info(f'{os.path.basename(path)}: {nr:,} lines, {len(found_triples):,}/{len(want_triples):,} '
                            f'wanted triples found')
                last = time.time()
    logger.info(f'{os.path.basename(path)}: {nr:,} lines in {time.time() - start:.0f}s; '
                f'{len(found_triples):,}/{len(want_triples):,} wanted triples in T_t, '
                f'{len(found_entities):,}/{len(want_entities):,} wanted entities in some triple')
    return found_triples, found_entities


def scan_entity_dictionary(path: str, want_entities: Set[str]) -> Set[str]:
    """Stream one entity dictionary once: the wanted entities it contains (`metadata.wikidata` of each line)."""
    found = set()
    start = last = time.time()
    nr = 0
    opener = gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8')
    with opener as f:
        for nr, line in enumerate(f, start=1):
            qid = json.loads(line)['metadata']['wikidata']
            if qid in want_entities:
                found.add(qid)
            if time.time() - last >= PROGRESS_SECS:
                logger.info(f'{path}: {nr:,} entities, {len(found):,}/{len(want_entities):,} wanted found')
                last = time.time()
    logger.info(f'{path}: {nr:,} entities in {time.time() - start:.0f}s; '
                f'{len(found):,}/{len(want_entities):,} wanted entities in the dictionary')
    return found


def load_kg_lookup(kg_snapshots_dir: str, entity_dicts_dir: str, snapshot_date: str,
                   want_triples: Set[Triple], want_entities: Set[str]) -> KgLookup:
    triples_in_t, entities_in_triples = scan_snapshot(snapshot_file(kg_snapshots_dir, snapshot_date),
                                                      want_triples, want_entities)
    in_dictionary = scan_entity_dictionary(entity_dictionary_file(entity_dicts_dir, snapshot_date), want_entities)
    return KgLookup(triples_in_t, entities_in_triples, in_dictionary)


def load_kg_lookups(kg_snapshots_dir: str, entity_dicts_dir: str,
                    wanted: Dict[str, Tuple[Set[Triple], Set[str]]]) -> Dict[str, KgLookup]:
    """{snapshot_date: (wanted triples, wanted entities)} -> {snapshot_date: KgLookup}; each file streamed once."""
    return {date: load_kg_lookup(kg_snapshots_dir, entity_dicts_dir, date, *wanted[date]) for date in sorted(wanted)}


def _flags(entry: Dict) -> Tuple[bool, bool]:
    eh, et = entry['emerging_head'], entry['emerging_tail']
    if not isinstance(eh, bool) or not isinstance(et, bool):
        raise TypeError(f'emerging_head / emerging_tail must be bools, got {eh!r} / {et!r} for {entry["triple"]}')
    return eh, et


def apply_rules_to_entry(entry: Dict, lookup: KgLookup, exists_in_instance: Set[Triple]):
    """decide_ops for one `tkgu_triples` entry (fields `triple`, `tkgu_operations`, `emerging_head/_tail`)."""
    key = triple_key(entry['triple'])
    eh, et = _flags(entry)
    return decide_ops(entry['tkgu_operations'], key, eh, et, lookup.in_t(key),
                      lookup.in_kg(key[0]), lookup.in_kg(key[2]), exists_in_instance)


def exists_of_instance(tkgu_triples: List[Dict], lookup: KgLookup) -> Set[Triple]:
    """The (s,p,o) of the instance's Exists operations that agree with the KG: a relabelled Add must not duplicate
    them. An Exists that the rules drop does not count, so its fact can still be kept through a relabelled Add."""
    kept = set()
    for t in tkgu_triples:
        if EXISTS in t['tkgu_operations']:
            key = triple_key(t['triple'])
            eh, et = _flags(t)
            verdict = op_violation(EXISTS, lookup.in_t(key), eh, et, lookup.in_kg(key[0]), lookup.in_kg(key[2]))
            if verdict == (None, None):
                kept.add(key)
    return kept
