#!/usr/bin/env bash
# Move the 47 GB per-cell MERFISH h5ad off the full root disk onto /mnt/data (14 TB),
# then symlink it back so any code path still resolves it. Run with sudo:
#     sudo bash scripts/move_big_h5ad.sh
set -euo pipefail

SRC=/home/fusar/lipidomics_tutorial_cajalcourse/C57BL6J-638850-imputed-log2.h5ad
DST=/mnt/data/cajal_lipidomics
USERN=fusar

mkdir -p "$DST"

if [ -f "$SRC" ] && [ ! -L "$SRC" ]; then
    echo "Moving $(du -h "$SRC" | cut -f1) -> $DST/ ..."
    mv "$SRC" "$DST"/
fi

chown "$USERN" "$DST"/C57BL6J-638850-imputed-log2.h5ad || true

if [ ! -e "$SRC" ]; then
    ln -s "$DST"/C57BL6J-638850-imputed-log2.h5ad "$SRC"
    chown -h "$USERN" "$SRC" || true
fi

echo "--- result ---"
ls -la "$SRC"
echo "--- root disk now ---"
df -h / | tail -1
