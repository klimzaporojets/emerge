"""Regression test for the file-iteration filter in the s05 entry point.

Background: job 22442903/22442904 (2026-05-03) wasted 4 H100-hours each
because `s05_generate_dataset_with_llm_v8.py:167` called `exit(0)` instead
of `continue` when the iteration encountered an input file not listed in
`input_files_to_process`. Result: the moment the OS-level file iteration
hit any unlisted file, the entire script terminated cleanly with exit 0,
producing zero output. The bug only triggered when sharded configs first
populated `input_files_to_process` with a non-empty list — the original
unsharded run had it as `[]` so the buggy branch was never taken.

This test AST-walks the entry point and asserts: every `if`-block inside
the file-iteration `for` loop that gates on `input_files_to_process`
exits via `continue`, not `exit / sys.exit / return / raise SystemExit`.
Cheap (<1 s), no GPU, no LLM, no I/O — it would have caught the original
bug before the sbatch ever ran.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "src" / "dataset" / "emerge" / "s05_generate_dataset_with_llm_v8.py"


def _find_input_files_to_process_filter(tree: ast.AST):
    """Yield every `if`-block whose test references `input_files_to_process`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names_in_test = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "input_files_to_process" in names_in_test:
            yield node


def _block_terminates_via(node: ast.If):
    """Return the AST class name of the early-exit statement in the if-body."""
    for stmt in node.body:
        if isinstance(stmt, ast.Continue):
            return "continue"
        if isinstance(stmt, ast.Return):
            return "return"
        if isinstance(stmt, ast.Raise):
            return "raise"
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Name) and func.id in ("exit", "quit"):
                return f"{func.id}()"
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in ("sys", "os")
                and func.attr in ("exit", "_exit")
            ):
                return f"{func.value.id}.{func.attr}()"
    return None


def test_input_files_filter_uses_continue():
    src = ENTRY.read_text()
    tree = ast.parse(src)
    matches = list(_find_input_files_to_process_filter(tree))
    assert matches, (
        f"No `if`-block referencing `input_files_to_process` found in {ENTRY}. "
        "The filter was either renamed or removed; update this test."
    )
    bad = []
    for node in matches:
        terminator = _block_terminates_via(node)
        if terminator != "continue":
            bad.append((node.lineno, terminator))
    assert not bad, (
        f"In {ENTRY}, the filter block(s) for `input_files_to_process` must "
        f"terminate via `continue` (skip-this-file), not `{bad}`. "
        "Using `exit()` / `sys.exit()` / `return` / `raise` here kills the "
        "entire script the moment iteration hits an unlisted file — the bug "
        "from jobs 22442903/22442904 (2026-05-03)."
    )
