#!/usr/bin/env bash
# Downloads the three result files that exceed GitHub's file limit from release v1.0.
set -e
cd "$(dirname "$0")"
BASE=https://github.com/tanvirul-1230/cccp-netload/releases/download/v1.0
[ -f "Additional_Analyses/final3/v3_tree/event_rows_test.csv.gz" ] || curl -fL --create-dirs -o "Additional_Analyses/final3/v3_tree/event_rows_test.csv.gz" "$BASE/v3_tree_event_rows_test.csv.gz"
[ -f "Additional_Analyses/final3/v3_mlp/event_rows_test.csv.gz" ] || curl -fL --create-dirs -o "Additional_Analyses/final3/v3_mlp/event_rows_test.csv.gz" "$BASE/v3_mlp_event_rows_test.csv.gz"
[ -f "Additional_Analyses/final3/v3_tree_clip/event_rows_test.csv.gz" ] || curl -fL --create-dirs -o "Additional_Analyses/final3/v3_tree_clip/event_rows_test.csv.gz" "$BASE/v3_tree_clip_event_rows_test.csv.gz"
echo "result files in place"
