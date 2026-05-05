#!/usr/bin/env bash
# run_all.sh — Run the full adversarial scenario experiment pipeline
# Usage:
#   bash run_all.sh              # full run (skips already-generated configs)
#   bash run_all.sh --regen      # re-generate everything from scratch
#   bash run_all.sh --figs-only  # only regenerate figures/report from saved data
set -euo pipefail
cd "$(dirname "$0")"

# ── Extract API keys from api_key.py ─────────────────────────────────────────
OPENAI_API_KEY=$(python3 -c "
import re, sys
m = re.search(r'OPENAI_API_KEY\s*=\s*(sk-[A-Za-z0-9\-_]+)', open('api_key.py').read())
sys.exit(0) if m else sys.exit(1)
print(m.group(1))
") || { echo "ERROR: OPENAI_API_KEY not found in api_key.py"; exit 1; }

ANTHROPIC_API_KEY=$(python3 -c "
import re, sys
m = re.search(r'ANTHROPIC_API_KEY\s*=\s*(sk-ant-[A-Za-z0-9\-_]+)', open('api_key.py').read())
sys.exit(0) if m else sys.exit(1)
print(m.group(1))
") || { echo "ERROR: ANTHROPIC_API_KEY not found in api_key.py"; exit 1; }

export OPENAI_API_KEY
export ANTHROPIC_API_KEY
echo "✓ OPENAI_API_KEY    loaded (${OPENAI_API_KEY:0:12}...)"
echo "✓ ANTHROPIC_API_KEY loaded (${ANTHROPIC_API_KEY:0:16}...)"
echo ""

# ── Parse flags ───────────────────────────────────────────────────────────────
REGEN=0
FIGS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --regen)     REGEN=1 ;;
    --figs-only) FIGS_ONLY=1 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ── Step 1: Baseline experiment (LLM-30 vs Random-30) ────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 1 — Baseline Experiment (GPT-4o vs Random, 30 each)"
echo "════════════════════════════════════════════════════════════════"
if [ $FIGS_ONLY -eq 1 ] || [ $REGEN -eq 0 ]; then
  python3 run_full_experiment.py --skip-gen --n 10
else
  python3 run_full_experiment.py --n 10
fi
echo ""

# ── Step 2: Multi-LLM comparison (all configs + Claude) ──────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 2 — Multi-LLM Comparison"
echo "  Configs: GPT-4o (std/enhanced/self-correct),"
echo "           GPT-4o-mini (std/enhanced),"
echo "           Claude Sonnet 4.6 (std/enhanced),"
echo "           Claude Haiku 4.5 (std)"
echo "════════════════════════════════════════════════════════════════"
if [ $FIGS_ONLY -eq 1 ]; then
  python3 run_multi_llm_experiment.py --figs-only
elif [ $REGEN -eq 1 ]; then
  python3 run_multi_llm_experiment.py --n 10 --no-skip
else
  python3 run_multi_llm_experiment.py --n 10
fi
echo ""

# ── Done ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Results:  results/multi_llm/          (scenario JSON per config)"
echo "  Figures:  figures/multi_llm/          (8 comparison figures)"
echo "            figures/eval_expanded/      (baseline comparison figures)"
echo "  Report:   results/evaluation_report.md"
echo ""
