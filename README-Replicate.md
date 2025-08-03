# Elevate3D on Replicate (Full Scaffold)

This folder contains the full version of the scaffold for Elevate3D with a complete predictor.

## Files
- `cog.yaml`: Builds PoissonRecon, downloads SAM checkpoint, installs requirements.
- `predict.py`: Full wrapper around Elevate3D's `main_refactor.py` including preprocessing fallback, mode toggles, and zipping outputs.
- `.github/workflows/replicate.yml`: CI workflow using `replicate/setup-cog@v2` to push the model to Replicate.

## Setup
1. Fork the official Elevate3D repo and add these files at the root.
2. In GitHub repository Settings → Secrets and variables → Actions, add:
   - `REPLICATE_CLI_AUTH_TOKEN`: your Replicate token
3. Trigger the workflow via GitHub Actions or push to `main`.

## Usage
- Input: `model_file` (.obj/.glb/.ply), optional `prompt_text`, `mode` (`full`/`texture_only`/`geometry_only`), etc.
- Output: ZIP of refinement results from `Outputs/3D/<dataset>/<obj_name>/...`.

> Adjust `predict.py` if upstream script names or flags change.
