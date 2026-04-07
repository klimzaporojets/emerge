# TODO

## Future work

- [ ] Upload ReLiK CIE entity indices (~121GB) to HuggingFace — needed only to re-run ReLiK CIE from scratch; pre-computed predictions are already in the test set
- [ ] Release full 233K dataset (currently only 3.5K test set)
- [ ] Rename scripts to clean names (e.g., `s0x_evaluate_predictions.py` -> `evaluate.py`)

## Code quality

- [ ] Add docstrings to remaining internal functions (scorers, preloader internals, dataset pipeline steps)
- [ ] Refactor duplicate `prepare_input_to_calculate` logic in `graph_matching.py` used by both `graph_scorers.py` and `entity_coverage_scorer.py`
- [ ] Clean up remaining commented-out code in `src/dataset/` pipeline files
- [ ] Replace remaining `print()` statements with `logger` calls in `src/dataset/` files
