#!/usr/bin/env bash
# bash reproduce.sh paper   numbers, tables, figures, checks and PDF from the included results (CPU, minutes)
# bash reproduce.sh full    everything from data/hems_data.zip (CUDA GPU, about 2.5 h); overwrites the results
set -e
cd "$(dirname "$0")"
export PYTHON=${PYTHON:-python}
bash fetch_results.sh
if [ "$1" = "full" ]; then
  (cd Additional_Analyses
   "$PYTHON" make_events3.py
   bash pipeline3.sh dev tree v3_tree
   bash pipeline3.sh test tree v3_tree
   bash pipeline3.sh test tree v3_tree_clip --clip-load --selection-tag v3_tree
   bash pipeline3.sh dev mlp v3_mlp
   bash pipeline3.sh test mlp v3_mlp
   FINAL2_ROOT="$(pwd)/final3" "$PYTHON" final_stats2.py v3_tree)
  (cd Paper/analysis && "$PYTHON" event_trace.py && "$PYTHON" battery_reserve.py)
  (cd Paper/verification_code && "$PYTHON" verify_lasting_loss.py)
elif [ "$1" != "paper" ]; then
  echo "usage: bash reproduce.sh paper|full"; exit 1
fi
(cd Paper/analysis && "$PYTHON" paper_numbers.py && "$PYTHON" paper_extra.py && "$PYTHON" battery_stats.py \
   && "$PYTHON" review1_numbers.py)
(cd Paper/figures && "$PYTHON" make_figures.py)
(cd Paper/verification_code && "$PYTHON" verify_paper_statements.py && "$PYTHON" verify_correction_guide.py)
(cd Paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
echo "Paper/main.pdf written"
