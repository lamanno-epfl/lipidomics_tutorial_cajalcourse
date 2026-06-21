#!/usr/bin/env bash
# Move a large file off the full root disk onto /mnt/data (14 TB) and symlink it back
# so code paths still resolve. Generic: pass the file path.
#     sudo bash scripts/move_to_mnt.sh /home/fusar/lipidomics_tutorial_cajalcourse/SOMEFILE
set -euo pipefail

SRC="${1:?usage: sudo bash scripts/move_to_mnt.sh <file>}"
SRC="$(readlink -f "$SRC" || echo "$SRC")"
DST=/mnt/data/cajal_lipidomics
USERN=fusar
BASENAME="$(basename "$SRC")"

mkdir -p "$DST"
if [ -f "$SRC" ] && [ ! -L "$SRC" ]; then
    echo "Moving $(du -h "$SRC" | cut -f1) -> $DST/ ..."
    mv "$SRC" "$DST"/
fi
chown "$USERN" "$DST/$BASENAME" || true
if [ ! -e "$SRC" ]; then
    ln -s "$DST/$BASENAME" "$SRC"
    chown -h "$USERN" "$SRC" || true
fi
echo "--- result ---"; ls -la "$SRC"
echo "--- root disk now ---"; df -h / | tail -1
