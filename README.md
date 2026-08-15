# PSI excitation energy transfer

Structure-based model of excitation energy transfer and trapping in photosystem I. Chlorophyll
positions and orientations are read from a PDB/mmCIF file, excitonic couplings are computed with
TrEsp transition charges, pigments are grouped into exciton domains, intra-domain relaxation is
treated with Redfield theory and inter-domain transfer with generalized Förster theory. A Pauli
master equation over the domains, with charge separation at P700 and a uniform loss channel, gives
the excitation lifetime and the trapping yield.

## Run

```bash
./run.sh                          # vibronic spectral density (default)
SPECTRAL_DENSITY=phonon ./run.sh  # low-frequency B777 spectral density only
```

Requires Python 3 with numpy, scipy and matplotlib. The bundled input is `1JB0.pdb`,
the photosystem I core of *Synechococcus elongatus* (96 chlorophylls).

## Spectral density

`SPECTRAL_DENSITY=vibronic` (default) uses the low-frequency B777 form of Renger & Marcus
together with 69 Chl *a* intramolecular modes above 100 cm⁻¹ taken from the difference
fluorescence-line-narrowing set of Reimers, Rätsep & Freiberg, *Front. Chem.* **8**, 588289 (2020),
Table 2. They enter the lineshape function g(t) as Franck–Condon sidebands only; the classical
reorganization shift stays low-frequency, since the site energies are effective 0–0 energies.

`SPECTRAL_DENSITY=phonon` keeps the B777 form alone.

The mode table in `chla_vibronic_TEA.txt` is reproduced from Table 2 of Reimers, Rätsep & Freiberg,
*Front. Chem.* **8**, 588289 (2020), published open access under CC BY 4.0; the file header carries
the citation.

**The transition dipole strength is switched with it.** Knox & Spring's tabulated 21.0 D² for
Chl *a* is the strength of the 0–0 transition, not of the whole Qy band. A coupling calibrated on
it therefore carries the factor e^(−S_vib) implicitly, which is exactly how a low-frequency-only
model accounts for the high-frequency vibrations (Friedl, Fedorov & Renger, *Phys. Chem. Chem.
Phys.* **24**, 5014 (2022)). Once those modes are put into g(t) explicitly, that factor has to be
removed, so the code uses 21.0/e^(−S_vib) = 30.5 D² in the vibronic case and 21.0 D² in the phonon
case. Do not mix the two.

## Parameters

| variable | default | meaning |
|---|---|---|
| `SPECTRAL_DENSITY` | `vibronic` | `vibronic` or `phonon` (see above) |
| `K_CS` | `0.667` | charge separation rate at P700, ps⁻¹ |
| `P700_E` | `14300` | P700 site energy, cm⁻¹ |
| `BELT_FUNNEL` | `0` | set to `1` to impose a site-energy gradient across antenna belts |
| `FUNNEL_STEP` | `50` | size of that gradient per belt, cm⁻¹, when enabled |
| `USE_TRESP` | `1` | TrEsp couplings; `0` falls back to the point-dipole approximation |

Set inside `psi_eet.py`: `V_CUTOFF = 60` cm⁻¹ for the exciton-domain partition, a Huang–Rhys factor
of 1 on the P700 chlorophylls and 0.5 elsewhere (distributed over excitons by participation), no
pure dephasing, and an initial excitation weighted by Qy dipole strength.

## Using another structure

Edit `PDB_FILENAME`, `RC_IDS` (chain and residue number of the two P700 chlorophylls) and
`CORE_CHAIN_IDS` at the top of `psi_eet.py`.

## Output

`run.sh` prints the spectral-density banner and the lifetime. The run also writes `domains.out`
(exciton domain composition), `population_dynamics.txt`, `excitation_decay.out` and flux summaries.

## License

MIT, see `LICENSE`. This covers the code in this repository. The chlorophyll mode table in
`chla_vibronic_TEA.txt` is reproduced under CC BY 4.0 from the reference given in its header, and
`1JB0.pdb` is the Protein Data Bank entry for the *Synechococcus elongatus* photosystem I core
(Jordan et al., *Nature* **411**, 909, 2001), redistributed under the PDB's terms of use.
