#!/usr/bin/env bash
# Test calculation: E. huxleyi PSI supercomplex (PDB 9JJ8, 563 chlorophylls) at the
# production configuration of the accompanying manuscript: binned site-energy landscape
# (10 A radial shells transferred from the plant PSI-LHCI landscape of Betti & Cupellini,
# JACS Au (2026), doi 10.1021/jacsau.6c00568), antenna Chl a at core mean + 51 cm^-1,
# S = 1 on the two P700 chlorophylls, k_CS = 2.8 ps^-1 on the lowest reaction-centre
# exciton, 4 ns intrinsic loss channel, static site-energy disorder of sigma = 100 cm^-1.
#
# usage: ./run_test.sh [N]     run N disorder realizations, seeds 1..N (default 1)
# One realization takes tens of minutes on one laptop core. The manuscript's production
# value is the mean over 500 realizations.
set -e
N=${1:-1}
PY=${PY:-python3}
export PDB_FILE=9JJ8.cif
export RC_IDS=a:801,b:806
export ANT_GAP=${ANT_GAP:-51}            # antenna Chl a offset above the core mean, cm^-1
export P700_E=0                          # keep the P700 site energies of site_energies.txt
export S_RC=${S_RC:-1}
export K_CS=${K_CS:-2.8}
export K_DISS_PS=${K_DISS_PS:-0.00025}   # 4 ns intrinsic loss
export DISORDER_FWHM=${DISORDER_FWHM:-235.5}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cp se_cocco.txt site_energies.txt
taus=""
for s in $(seq 1 "$N"); do
  export DISORDER_SEED=$s
  LOG=run_s$s.log
  "$PY" -u ../eng.py > "$LOG" 2>&1
  if [ "$s" = "1" ]; then
    grep -m1 '\[RC-CHECK\]' "$LOG"
    grep -m1 '\[DISORDER\]' "$LOG"
  fi
  t=$(grep -o 'tau = [0-9.]*' "$LOG" | tail -1 | cut -d' ' -f3)
  echo "seed $s   tau = $t ps"
  taus="$taus $t"
done
if [ "$N" -gt 1 ]; then
  echo "$taus" | "$PY" -c "
import sys
v = [float(x) for x in sys.stdin.read().split()]
m = sum(v) / len(v)
sd = (sum((x - m)**2 for x in v) / (len(v) - 1))**0.5
print('mean over %d realizations: %.3f +- %.3f ps (sd %.2f)' % (len(v), m, sd / len(v)**0.5, sd))
"
fi
