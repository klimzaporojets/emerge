# Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction

> ## ⚠️ EMERGE users — read this first
>
> **This wrapper adapts the upstream [EDC repository](https://github.com/clear-nus/edc).** The upstream `datasets/tt2kg/{kgs/, passages/}` data layout it uses **does not apply to EMERGE.**
>
> For EMERGE, the actual entry point is [`s01_run_v3.py`](s01_run_v3.py) in this directory, which reads from EMERGE's standard data layout produced by `./scripts/download_data.sh`:
>
> 1. **Download the data** from the repo root:
>    ```bash
>    ./scripts/download_data.sh --indices
>    ```
>    This fetches `data/indices/relik_edc_relation_indexes/` (~400 MB) which `s01_run_v3.py` hard-asserts. **Without `--indices` you'll get an `AssertionError` on `target_schema_path`** (this was [issue #1](https://github.com/klimzaporojets/emerge/issues/1)).
> 2. The eval set passages already live at `data/evaluation_set/snapshot_*/delta_*.jsonl` after the default `./scripts/download_data.sh` (no extra flag needed).
> 3. Run the wrapper through the standard EMERGE benchmark orchestrator:
>    ```bash
>    ./scripts/run/run_benchmark.sh config/benchmarks/s02_run_benchmarks/<edc-plus-config>/config.json
>    ```
>
> See [`src/benchmarks/README.md`](../../../README.md) and the top-level [`README.md`](../../../../../README.md) for the full benchmark workflow.

---

## Credit

This EMERGE wrapper builds on the **EDC** framework (Extract, Define, Canonicalize) by Bowen Zhang and Harold Soh. The full upstream documentation — its own `datasets/tt2kg` data layout, `run.py` options, and schema-retriever training — lives in the upstream repository and does **not** apply to EMERGE; follow the EMERGE workflow above instead.

- Paper: [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction](https://arxiv.org/abs/2404.03868)
- Upstream repository: https://github.com/clear-nus/edc

Please cite EDC:

```bibtex
@article{zhang2024extract,
  title={Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction},
  author={Zhang, Bowen and Soh, Harold},
  journal={arXiv preprint arXiv:2404.03868},
  year={2024}
}
```
