# Python ↔ Rust parity harness

Side-by-side comparison tooling against the Python reference
implementation in `/Users/sasha/Desktop/spektrafilm`.

## One-time setup

Python 3.13 + a venv with the spektrafilm reference deps:

```bash
brew install python@3.13
/opt/homebrew/bin/python3.13 -m venv /tmp/spektravenv
/tmp/spektravenv/bin/pip install --upgrade pip setuptools wheel
/tmp/spektravenv/bin/pip install --prefer-binary numpy scipy colour-science \
    scikit-image opt-einsum lmfit Pillow numba rawpy PyYAML OpenImageIO \
    exiv2 pyfftw lensfunpy
/tmp/spektravenv/bin/pip install --no-deps -e /Users/sasha/Desktop/spektrafilm
```

## Scripts

- `spektra_compare.py <raw.dng> <out_dir> [film] [paper] [max_dim]`
  Decodes the RAW via rawpy → linear ACES2065-1, downsamples to
  `max_dim` (default 1500 px on the long edge) so Python doesn't OOM at
  full resolution, saves the downsampled image as a linear-float TIFF,
  and renders it through the Python pipeline. Prints the suggested Rust
  CLI invocation to render the same TIFF.

- `py_stages.py`
  Per-stage dump for 5 hard-coded test colors. Prints `raw`, `log_raw`,
  `film dens`, and `final RGB` after running through Python's
  SimulationPipeline. Pair with the `rs_stages` probe binary (kept in
  `/tmp/cmp/rs_stages.rs` during the diagnostic session) for a stage-by-
  stage comparison.

## Current findings (session ending 2026-05-23)

Tested on `IMG_8096.dng` downsampled to 1500 px, kodak_gold_200 +
fujifilm_crystal_archive_typeii, all effects off, AE off:

| Stage                                | Match status                  |
|--------------------------------------|-------------------------------|
| `build_rgb_to_adapted_xyz` matrix    | bit-identical to 7 decimals   |
| Hanatos2025 spectral upsampling       | bit-identical to 6 decimals   |
| log10 + film density curve interp    | bit-identical                 |
| Scan-film mode (skip print + scan)   | 1/255 quantization only       |
| Full chain with printing             | mean 5/255 drift, max 14/255  |

**Conclusion:** the drift is entirely in the **print stage** (between
the film density CMY output and the print density CMY output). The most
likely culprits are:

- `apply_database_neutral_print_filters` neutral filter values
- `compute_exposure_factor` print exposure normalization
- The spectral integration through the film dyes + enlarger illuminant
  + print sensitivities
- The print density curve interpolation

Verified the c/m/y neutral filter values match Python exactly:
`0.0 / 76.95409743164727 / 82.69937618751322` for kodak_gold_200 +
fujifilm_crystal_archive_typeii + TH-KG3.
