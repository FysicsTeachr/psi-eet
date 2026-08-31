# PSI excitation energy transfer

Structure-based model of excitation energy transfer and trapping in photosystem I supercomplexes.
Chlorophyll positions and orientations are read from a PDB/mmCIF file, excitonic couplings are
computed with TrEsp transition charges, pigments are partitioned into exciton domains by
participation-ratio clustering, intra-domain relaxation is treated with Redfield theory
(per-pigment spectral densities J_m = S_m J_ph), inter-domain transfer with generalized Förster
theory, and detailed balance is enforced on the assembled rate matrix. A master equation over all
exciton states, with charge separation on the lowest exciton of the reaction-centre domain and a
configurable intrinsic loss channel, gives the excitation lifetime and trapping yield.
High-frequency intramolecular Chl modes enter the line shapes as Franck–Condon sidebands (the mode
table of Reimers, Rätsep & Freiberg, *Front. Chem.* **8**, 588289 (2020), CC BY 4.0), and the
transition charges are scaled accordingly to the full Qy dipole strength.

The engine is `eng.py` (Python 3 + numpy + scipy; no other dependencies). All model choices are
set through environment variables; the test below sets every one that the production calculations
of the accompanying manuscript use.

## Test calculation: E. huxleyi PSI at the production configuration

`test_ehuxleyi_bsb/` contains everything needed to rerun the largest system of the manuscript:
the *E. huxleyi* PSI supercomplex (PDB 9JJ8, 563 chlorophylls) with the binned site-energy
landscape (10 Å radial shells transferred from the plant PSI-LHCI landscape of Betti & Cupellini,
*JACS Au* (2026), doi 10.1021/jacsau.6c00568; antenna Chl a at core mean + 51 cm⁻¹), S = 1 on the
two P700 chlorophylls, k_CS = 2.8 ps⁻¹ on the lowest reaction-centre exciton, a 4 ns intrinsic
loss channel, and static site-energy disorder of σ = 100 cm⁻¹. The script takes the number of
disorder realizations as its argument (seeds 1..N; default 1) and prints each realization's
lifetime plus, for N > 1, their mean:

```bash
cd test_ehuxleyi_bsb
./run_test.sh        # one realization (seed 1)
./run_test.sh 3      # seeds 1-3 and their mean
```

Expected lifetimes (single core, tens of minutes per realization; last digits may vary by
~0.001 ps between BLAS builds):

| DISORDER_SEED | tau (ps) |
|---|---|
| 1 | 113.344240 |
| 2 | 108.022344 |
| 3 | 89.804785 |

Individual realizations scatter widely (sd ≈ 7 ps); the production value is the mean over seeds
1–500, **103.1 ± 0.3 ps** (and 88.534 ps with disorder switched off, `DISORDER_FWHM=0`).

## Legacy

`legacy/` holds the previous version of this code (`psi_eet.py`, single-structure demo on the
*T. elongatus* core). It predates the revised model and is kept only for the record — do not use
it to reproduce the manuscript.
