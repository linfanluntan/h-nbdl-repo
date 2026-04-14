#!/bin/bash
# Reproduce all results from the H-NBDL paper.
#
# Usage:
#   bash scripts/reproduce_paper.sh          # full pipeline
#   bash scripts/reproduce_paper.sh --quick  # synthetic only, fewer seeds
set -e

QUICK=false
if [[ "$1" == "--quick" ]]; then QUICK=true; fi

echo "============================================"
echo "  H-NBDL: Reproducing Paper Results"
echo "============================================"

mkdir -p results figures

# ── 1. Synthetic Benchmarks (Section 6.1) ──
echo ""
echo "── [1/6] Synthetic Benchmarks ──"
python experiments/synthetic/run_synthetic.py \
    --config configs/synthetic.yaml \
    --output results/synthetic_results.json

# ── 2. Multi-Site Radiomics (Section 6.2) ──
echo ""
echo "── [2/6] Multi-Site Radiomics ──"
python experiments/radiomics/run_radiomics.py \
    --config configs/radiomics.yaml \
    --output results/radiomics_results.json \
    2>/dev/null || echo "  Skipped: data not available (see README)."

# ── 3. Cardiac EP (Section 6.3) ──
echo ""
echo "── [3/6] Cardiac Electrophysiology ──"
python experiments/electrophysiology/run_ep.py \
    --config configs/ep.yaml \
    --output results/ep_results.json \
    2>/dev/null || echo "  Skipped: data not available (see README)."

# ── 4. Ablation Studies (Section 6.4) ──
echo ""
echo "── [4/6] Ablation Studies ──"
if $QUICK; then
    echo "  (Quick mode: reduced seeds)"
    SEEDS=2; EPOCHS=50
else
    SEEDS=5; EPOCHS=100
fi
python -c "
from h_nbdl.analysis import run_pooling_ablation, run_kmax_ablation, run_temperature_ablation
import json
results = {}
results['pooling'] = [{'param': r.param_value, 'amari': r.amari} for r in run_pooling_ablation(n_seeds=${SEEDS}, epochs=${EPOCHS})]
fixed, ibp = run_kmax_ablation(n_seeds=${SEEDS}, epochs=${EPOCHS})
results['kmax'] = {'fixed': [{'K': r.param_value, 'amari': r.amari} for r in fixed], 'ibp_amari': ibp.amari}
results['temperature'] = [{'tau': r.param_value, 'mse': r.recon_mse} for r in run_temperature_ablation(n_seeds=${SEEDS}, epochs=${EPOCHS})]
with open('results/ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('  Ablation results saved.')
"

# ── 5. Proposition Verification (Sections 3.7, 4.4, 5.2) ──
echo ""
echo "── [5/6] Verifying Theoretical Propositions ──"
python -m h_nbdl.analysis.verify_propositions 2>&1 | tee results/proposition_verification.txt

# ── 6. Generate Paper Figures ──
echo ""
echo "── [6/6] Generating Paper Figures ──"
python scripts/figures/paper_figures.py 2>/dev/null && echo "  Figures saved to figures/"

echo ""
echo "============================================"
echo "  All experiments complete."
echo "  Results: results/"
echo "  Figures: figures/"
echo "============================================"
