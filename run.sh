#!/usr/bin/env bash
# Minimal driver: run the model on the bundled structure and print the excitation lifetime.
#
#   ./run.sh                          # default: vibronic spectral density
#   SPECTRAL_DENSITY=phonon ./run.sh  # low-frequency B777 only
#
set -e
PY=${PY:-python3}
export SPECTRAL_DENSITY=${SPECTRAL_DENSITY:-vibronic}   # vibronic | phonon
export K_CS=${K_CS:-0.667}          # charge separation rate, ps^-1
export P700_E=${P700_E:-14300}      # P700 site energy, cm^-1
export BELT_FUNNEL=${BELT_FUNNEL:-0}
export USE_TRESP=${USE_TRESP:-1}
LOG=${LOG:-run.log}
"$PY" -u psi_eet.py > "$LOG" 2>&1
echo "spectral density : $SPECTRAL_DENSITY"
grep -h '^\[SD\]' "$LOG"
grep -h 'Overall Lifetime\|Overall Yield' "$LOG" | sed 's/^ *//'
