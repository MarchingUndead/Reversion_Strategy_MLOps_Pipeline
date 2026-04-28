#!/usr/bin/env bash
# Build report/report.tex into report/report.pdf via latexmk.
#
# Requires a TeX Live / MiKTeX installation with latexmk + pdflatex.
# On Windows: MiKTeX includes both. On Debian/Ubuntu:
#   sudo apt install texlive-latex-recommended texlive-latex-extra latexmk
#
# Usage:
#   bash scripts/build_report.sh          # one-shot build
#   bash scripts/build_report.sh --watch  # live rebuild on save
#
set -euo pipefail
cd "$(dirname "$0")/.."

TEX="report/report.tex"
OUT="report/_build"
mkdir -p "$OUT"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "ERROR: latexmk not found. Install TeX Live (Linux/macOS) or MiKTeX (Windows)." >&2
  exit 1
fi

case "${1:-}" in
  --watch|-w)
    echo "[build_report] watching $TEX (Ctrl-C to stop)..."
    latexmk -pdf -pvc -interaction=nonstopmode -outdir="$OUT" "$TEX"
    ;;
  *)
    echo "[build_report] one-shot build..."
    latexmk -pdf -interaction=nonstopmode -outdir="$OUT" "$TEX"
    cp "$OUT/report.pdf" "report/report.pdf"
    echo "[build_report] -> report/report.pdf"
    ;;
esac
