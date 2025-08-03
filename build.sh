#!/usr/bin/env bash
set -euo pipefail

echo "=== inside container, pwd ==="
pwd
echo "=== listing root ==="
ls -al .

# fail fast if missing
if [ ! -f requirements.txt ]; then
  echo "🛑 requirements.txt not found in expected location"
  exit 1
fi

# Python deps
python -m pip install --upgrade pip
pip install -r requirements.txt cog

# SAM checkpoint
mkdir -p Checkpoints/sam
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P Checkpoints/sam/

# Build PoissonRecon
git clone https://github.com/mkazhdan/PoissonRecon.git || true
make -C PoissonRecon
