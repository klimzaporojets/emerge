"""Static signature audit — catches arg-count / arg-name mismatches between
callers and callees without running any LLM or loading the GPU.

Background: job 22422141 (2026-05-02 1-h smoke test) burnt 1 H100-hour on 4,148
identical TypeErrors because `s05_prompt_llm_utils_v8.py` calls
`get_prompt_from_config()` with 5 args, but the def in `s05_prompts.py` was
extended to require 7. This test would have caught it in <1 second on a
laptop with no GPU.

Strategy: for each module in the audit set, AST-walk every `Call` node whose
`.func` is a bare `Name` matching a function imported into that module. Look up
the callee's `inspect.signature` and try `sig.bind(*sentinels, **sentinels)`. If
binding raises TypeError, the call is broken — report the file/line.

Scope is intentionally narrow to start (the v8 prompt-LLM caller pair). Add
more (caller_module, callee_module) pairs to AUDIT_PAIRS as the codebase grows.
"""
import ast
import importlib
import inspect
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


# Each pair: (caller_file_relative_to_src, callee_module_dotted_path)
AUDIT_PAIRS: List[Tuple[str, str]] = [
    # The bug from job 22422141 lives in this pair.
    (
        "dataset/emerge/utils/s05_prompt_llm_utils_v8.py",
        "dataset.emerge.utils.s05_prompts",
    ),
    # Parent caller of the prompt-LLM utils.
    (
        "dataset/emerge/utils/s05_generate_dataset_utils_v8.py",
        "dataset.emerge.utils.s05_prompt_llm_utils_v8",
    ),
    # Top-level entrypoint imports the utils.
    (
        "dataset/emerge/s05_generate_dataset_with_llm_v8.py",
        "dataset.emerge.utils.s05_generate_dataset_utils_v8",
    ),
    # Parallel `get_prompt_from_config` lives in misc/llms — different caller chain,
    # but same shape, so audit it too to prevent the regression from creeping in.
    (
        "evaluation/scorers/misc/llm_calls.py",
        "misc.llms.llm_prompting",
    ),
]


def _collect_calls(caller_path: Path):
    """Return a list of (lineno, name, n_positional, [kw_names]) for every
    Call node whose .func is a bare Name. Starred / **kwargs unpacks are
    skipped (we cannot statically check those)."""
    tree = ast.parse(caller_path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        # Skip if call uses *args or **kwargs (cannot statically resolve)
        has_star = any(isinstance(a, ast.Starred) for a in node.args)
        has_dstar = any(k.arg is None for k in node.keywords)
        if has_star or has_dstar:
            continue
        kw_names = [k.arg for k in node.keywords]
        out.append((node.lineno, node.func.id, len(node.args), kw_names))
    return out


def _audit_pair(caller_rel: str, callee_module: str):
    caller_path = SRC / caller_rel
    assert caller_path.is_file(), f"caller not found: {caller_path}"

    mod = importlib.import_module(callee_module)
    callees = {
        name: obj
        for name, obj in vars(mod).items()
        if callable(obj) and not name.startswith("_")
    }

    failures = []
    for lineno, name, n_pos, kw_names in _collect_calls(caller_path):
        if name not in callees:
            continue  # not a same-module function
        sig = inspect.signature(callees[name])
        # Sentinels for static binding check
        sentinel_pos = [object()] * n_pos
        sentinel_kw = {k: object() for k in kw_names}
        try:
            sig.bind(*sentinel_pos, **sentinel_kw)
        except TypeError as e:
            failures.append(
                f"  {caller_rel}:{lineno}  {name}({n_pos} pos + kw={kw_names})  "
                f"→ {e}  (def: {callees[name].__module__}.{name}{sig})"
            )
    return failures


def test_signature_audit():
    all_failures = []
    for caller_rel, callee_module in AUDIT_PAIRS:
        failures = _audit_pair(caller_rel, callee_module)
        all_failures.extend(failures)
    if all_failures:
        msg = (
            f"\n{len(all_failures)} signature mismatch(es) found across "
            f"{len(AUDIT_PAIRS)} audit pair(s):\n" + "\n".join(all_failures)
        )
        raise AssertionError(msg)
