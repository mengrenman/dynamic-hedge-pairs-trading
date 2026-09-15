# Notebook build scripts

Two notebooks are generated from Python rather than edited by hand, so their code and prose can be
diffed and reviewed like source:

| script | builds | from |
|---|---|---|
| `build_tv_notebook.py` | `../tv_cointegration_kalman.ipynb` | scratch |
| `build_tuning_notebook.py` | `../pairs_trading_04_hyperparameter_tuning.ipynb` | `../pairs_trading_02.ipynb` (inherits every cell it does not rewrite) |
| `build_viz_notebook.py` | `../visualize_cointegrated_pairs.ipynb` | scratch |

Regenerate a notebook (cells only, outputs cleared) and then execute it:

```bash
python notebooks/build/build_tv_notebook.py
python notebooks/build/execute.py notebooks/tv_cointegration_kalman.ipynb

python notebooks/build/build_tuning_notebook.py
python notebooks/build/execute.py notebooks/pairs_trading_04_hyperparameter_tuning.ipynb
```

Both builders write the target notebook in place; pass `--out PATH` to write elsewhere (for example to
diff against the committed version). `execute.py` runs with `notebooks/` as the working directory,
stores the outputs, and exits non-zero if a cell raised.

Notes

- Executing needs the package installed with the `notebooks` extra and network access for OpenBB; prices
  and the screening table are cached under `notebooks/cache/` (gitignored) after the first run.
- On a 16-core machine the time-varying cointegration notebook takes about 3 minutes and the tuning
  notebook about 8 minutes from a cold cache (2 minutes warm).
- Because `build_tuning_notebook.py` starts from `pairs_trading_02.ipynb`, a change to nb02's pipeline
  cells propagates to nb04 by re-running the builder; the cells it rewrites are asserted on content, so
  the script fails loudly if nb02 drifts.
