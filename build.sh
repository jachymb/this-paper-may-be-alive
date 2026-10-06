#!/usr/bin/env bash
# Builds the paper.  ./build.sh --sample  -> sample.pdf (front matter + position 1-2 / layer 1 + a truncated LM head; ~3 min)
#                    ./build.sh           -> main.pdf   (everything: 95,225 pages, 814 MB; generation ~70 s, ~30 min per pdflatex pass)
set -euo pipefail
cd "$(dirname "$0")"
uv run --quiet generate.py "$@"
if [[ "${1:-}" == "--sample" ]]; then
  job=sample; extra='\def\SAMPLE{1}'
else
  job=main; extra=''
fi
run() { pdflatex -interaction=batchmode -halt-on-error -jobname="$job" "$extra\\input{main}" >/dev/null || { tail -40 "$job.log"; exit 1; }; }
echo "pass 1 ($(date +%H:%M))"; run
bibtex "$job" >/dev/null || true
echo "pass 2 ($(date +%H:%M))"; run
echo "pass 3 ($(date +%H:%M))"; run
echo "done ($(date +%H:%M))"
grep -h "Output written" "$job.log"
echo "overfull boxes: $(grep -c Overfull "$job.log" || true)"
grep -i "warning" "$job.log" | grep -v "Overfull\|Underfull" | sort | uniq -c | head
