#!/usr/bin/env bash
# Protocol v3, one process per household (peak memory), then merge.
#   bash pipeline3.sh dev  tree v3_tree
#   bash pipeline3.sh test tree v3_tree [extra run3 args, e.g. --clip-load --selection-tag v3_tree]
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
STAGE=$1; BB=$2; TAG=$3; shift 3
SELTAG=$TAG
for a in "$@"; do [ "$prev" = "--selection-tag" ] && SELTAG=$a; prev=$a; done
for H in H3 H4 H6; do
  "$PY" run3.py --stage "$STAGE" --backbone "$BB" --tag "$TAG/$H" --households "$H" --selection-tag "$SELTAG" "$@"
done
"$PY" merge3.py "$TAG" "$STAGE"
if [ "$STAGE" = "dev" ]; then "$PY" select_params3.py "$TAG"; fi
echo "PIPELINE $STAGE $TAG FINISHED"
