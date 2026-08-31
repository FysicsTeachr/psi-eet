# =============================================================================
#  EET engine  v5_quadfix
#  Derived from v4_quadfix/eng.py (md5 c572fa12f32031d0466114f43ac3dc1d, 2026-08-22 02:42)
#  by the 2026-08-22 audit.  PHYSICS DEFAULTS ARE UNCHANGED except FIX 1.
#
#   FIX 1  Forster lineshape S weighting: Sum_i c_ia^4 S_i, not gamma_aa * S_a.
#          CHANGES NUMBERS (only for excitons mixing P700 S=1 with antenna S=0.5).
#          V4_LINESHAPE_COMPAT=1 restores v4 bit-for-bit.
#   FIX 2  [ENGINE]/[SETTINGS] provenance banner + provenance.json.
#   FIX 3  [EXCITATION] label corrected (v4 said "non-core antenna"; it is all chains).
#   FIX 4  [DOMAINS] names the partition actually used and prints PR_SEED.
#   FIX 5  [DB-ENFORCE] counts pairs modified, not pairs visited.
#   FIX 6  db_dump.npz written once (v4 wrote it twice, second dropping fields).
#   FIX 7  [K-CLIP] the 500 ps^-1 ceiling is reported instead of silent.
#   FIX 8  [LEGACY-V1] the v1 domain-level outputs are labelled as such.
#  FIX 2-8 are diagnostics/labels only and cannot move a number.
#
# =============================================================================
#  SETTLED - the defaults below ARE the production model.  `python3 eng.py` with an
#  empty environment reproduces it.  DO NOT REOPEN THESE:
#
#   VIBRONIC=1           high-frequency modes are required.  Renger, JPCB 2021: for large
#                        gaps between chemically distinct pigments the acoustic phonon SD
#                        is not enough.  Without them the 986 cm-1 Chl a<->Chl c channel
#                        is 62x too slow.
#   LOC_VIB=1            the IPR weight gamma_aa applies to the PHONON only, not to the
#                        intramolecular modes.  Required for consistency with the per-site
#                        dipole: for localised modes the exciton 0-0 carries e^-S_vib, not
#                        e^-(gamma_aa*S_vib).  = Saraceno's "cR loc. vib.", JCP 164, 044119.
#   LINESHAPE_GAMMA=1    KEEP the lifetime broadening Re(gamma_a) in the Forster lineshapes.
#                        Kim, Akhtar, Betti and Saraceno all do.  Detailed balance is exact
#                        either way once DB_ENFORCE is on, so there is nothing to buy by
#                        removing it.  ** Setting this to 0 is BANISHED - do not resurrect. **
#   REDFIELD_RENORM_E=0  Redfield rates at the BARE Bohr frequency, as Betti/pyQME.  The
#                        w-tilde substitution was our own departure, is redundant under
#                        enforcement, and was not claimed in the submitted manuscript.
#   DB_ENFORCE=1         downhill-anchored detailed balance on the E-tilde ladder.  This is
#                        Betti's own enforce_detailed_balance_rates, and it is what answers
#                        Reviewer 1 point 7.
#   dipole convention    mu^2 = 21.0 * e^S_vib, COMPUTED from the loaded mode table, and the
#                        TrEsp charge scale and the Chl c dipole ride on it automatically.
#                        21.0 D^2 is Knox & Spring's 0-0 value (they say so explicitly);
#                        Madjet's charges are rescaled to exactly that (4.6 D).
#
#   S_RC=2.6             Huang-Rhys factor of the phonon bath on the two P700 Chls, giving
#                        lambda_RC = 204 cm-1.  Mid-range of the ONLY P700-specific
#                        measurement (Gillie 1989 hole burning: lambda = 140-300 cm-1).
#                        The previous 1.0 had no published source.  Full note at the
#                        HUANG_RHYS_S_RC definition.
#   VIB_WMIN=300         phonon / intramolecular-mode boundary.  Three anchors, documented
#                        in full at the mode loader: Ratsep's ~90 cm-1 overlap feature, the
#                        ~260 cm-1 crossover measured from this engine's own two components,
#                        and Ratsep's ~300 cm-1 "precluding an unambiguous separation".
#                        The whole boundary question spans only 67.3-71.2 ps (+-2.8%).
#
#  The only genuinely open parameter is PR_CUT, and at VIB_WMIN=300 it spans just +-0.55%.
# =============================================================================
import numpy as np
import os
import random
from math import sqrt, log, cos, pi, factorial, exp
import numpy.linalg as linalg
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from collections import defaultdict
import shutil
from scipy.linalg import expm
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt

# ==============================================================================
#  1. PHYSICAL CONSTANTS
# ==============================================================================
HBAR = 5308.738       # cm^-1 * fs / rad
KB   = 0.6949948      # cm^-1 / K

# ==============================================================================
#  2. SIMULATION PARAMETERS
# ==============================================================================
TEMP_K = 300.0

# Integration Grid
FREQ_STEP_CM = 0.5
FREQ_MAX_CM  = 2000.0
TIME_STEP_FS = float(__import__('os').environ.get('TIME_STEP_FS', '1.0'))
_CLAMP = {'neg': 0, 'hi': 0, 'negmin': 0.0, 'n': 0}
_REORG_PARTS = []
_DISP_PARTS = []
TIME_MAX_FS  = 5000.0

# Renger Parameters
RENGER_S1 = 0.8
RENGER_S2 = 0.5
RENGER_W1_CM = 0.56
RENGER_W2_CM = 1.94

# --- MODIFIED HUANG RHYS SECTION ---
HUANG_RHYS_S_CORE = 0.5
HUANG_RHYS_S_ANT  = float(__import__('os').environ.get('S_ANT', '0.5'))  # bulk antenna phonon S
# (0.5 is the B777/Renger-Marcus value.  Gillie 1989 quote S ~ 0.8 for PSI-200 antenna
#  Chl a/b; Adolphs 2010 fit S = 0.6 for all 96 Chls.  Knob added for sensitivity only.)
HUANG_RHYS_S_RED = 0.5   # v1 legacy grid lineshape only
# --- S_REDX: per-pigment Huang-Rhys for RED forms in the EXCITON path.
# Default "" = inert, reds keep S_ANT exactly as before.  Set to a number to give
# every pigment flagged is_red (i.e. listed in site_energies.txt) its own S.
# ** Transfer lambda, NOT S. **  Khmelnitskiy 2020 JPCB 124,8504 quote S=3-5 for the
# Synechocystis red states, but tied to their own (much lower) mean phonon frequency.
# Their Table 2 Stokes shifts are the transferable quantity: lambda = Delta/2 gives
# C706 59.5, C714 75.0, C710 146.0 cm-1  ->  S = lambda/78.535 = 0.76 / 0.96 / 1.86
# in this engine's phonon convention.  Bulk antenna is S_ANT=0.5 -> lambda 39.3 cm-1.
_SREDX = __import__('os').environ.get('S_REDX', '')
HUANG_RHYS_S_REDX = (float(_SREDX) if _SREDX.strip() != '' else None)
# Huang-Rhys factor of the PHONON bath on the two P700 Chls.  SETTLED at 2.6 (2026-08-22).
#
#  The ONLY P700-specific measurement is Gillie, Lyle, Small & Golbeck, Photosynth. Res. 22,
#  233 (1989), "Spectral hole burning of the primary electron donor state of Photosystem I",
#  verbatim: "A Huang-Rhys factor S in the range 4-6 and a corresponding mean phonon
#  frequency in the range 35-50 cm-1 ... The zero-point level of P700* is predicted to lie
#  at ~710 nm at 1.6 K with an absorption maximum at ~702 nm."  They also state
#  "S omega_m is the reorganization energy", giving lambda = 140-300 cm-1, which is the
#  ~200 cm-1 that Madjet 2009 quotes for the special pair.
#
#  ⛔ S = 4-6 is NOT transferable: S and omega_m trade off in their fit, and our B777 phonon
#  has an effective mean frequency of 78.5 cm-1 against their 35-50.  Transfer LAMBDA, not S.
#  lambda_RC = 78.535 * S_RC, so S_RC = 2.6 -> 204.2 cm-1, mid-range of the measurement.
#
#  ⛔ The old default of 1.0 had NO published source: no PSI exciton model assigns the RC a
#  Huang-Rhys factor different from the bulk antenna.  Adolphs, Mueh, Madjet, Schmidt am
#  Busch & Renger, JACS 132, 3331 (2010) use one spectral density for all 96 Chls ("the
#  Huang-Rhys factor S = int dw J(w) ... to be S = 0.6 ... E_lambda = 35 cm-1") and never
#  mention Gillie or hole burning.  Byrdin 2002 has no Huang-Rhys factor at all.
#  So 0.5-0.6 is what every model does, 1.0 was ours alone, 2.6 follows the measurement.
HUANG_RHYS_S_RC = float(__import__('os').environ.get('S_RC', '2.6'))
# -----------------------------------
RED_FORM_DISTANCE_CUTOFF = 8.8  # Angstroms (Mg-Mg distance for strong coupling)
# Controls
N_MONTE_CARLO_RUNS = 1
# Structure file and the two P700 chlorophylls.  Defaults are the diatom, so the
# diatom result is unchanged.  Other species (all on HPCC in eet_kcs/<sp>/):
#   diatom    PDB_FILE=experimental.pdb   RC_IDS=a:803,b:804
#   cmerolae  PDB_FILE=5ZGB.pdb           RC_IDS=A:801,B:806
#   igalbana  PDB_FILE=8Z11.pdb           RC_IDS=a:801,b:805
#   species4  PDB_FILE=6ly5-aligned.pdb   RC_IDS=a:801,b:805
#   cocco     PDB_FILE=9JJ8.cif           RC_IDS=a:801,b:806   (mmCIF)
#   cyano     PDB_FILE=1JB0.pdb           RC_IDS=A:1011,B:1021
PDB_FILENAME = __import__("os").environ.get("PDB_FILE", "experimental.pdb")
SITE_ENERGY_FILENAME = "site_energies.txt"
DOMAIN_FILE = "domains.out"

INITIAL_EXCITATION_CHAINS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'}  # pentamer-like (ChimeraX overlap with Euglena USPTY pentamer)

# Structure Params
# Chl a Qy dipole.  Knox & Spring, Photochem. Photobiol. 77, 497 (2003), Table 2:
# D0(cavity) = 21.0 D^2, and they state explicitly that their values are the '0-0'
# strengths.  Madjet 2006 rescale their TrEsp charges to exactly this (4.6 D).
# With explicit high-frequency modes the coupling must be un-renormalised by e^S_vib
# (Caycedo-Soler, Nat. Commun. 13, 2912 (2022); Saraceno, JCP 164, 044119 (2026)).
# ==> recomputed from the LOADED mode table just below, so the two can never drift apart.
MU2_CHLA_00 = 21.0
MAGNITUDE_MU_A = sqrt(MU2_CHLA_00)
REFRACTIVE_INDEX = 1.4
VMN_PREFACTOR = 5.04
C_FACTOR = (REFRACTIVE_INDEX**2 + 2.0)**2 / (9.0 * REFRACTIVE_INDEX**2)
# Static site-energy disorder on the Hamiltonian used for the RATES.  Default 0 keeps
# the historical single-realization behaviour.  Betti & Cupellini SI: "we generated two
# separate ensembles of 500 Hamiltonians ... by sampling each site energy from a gaussian
# distribution centered on the computed site energy and with standard deviation
# sigma_LE = 100 cm-1" -> DISORDER_FWHM = 235.5.
# NOTE the DOMAIN PARTITION is NOT affected: it comes from the disorder-averaged
# participation-ratio matrix (PR_NREAL realizations at PR_FWHM), which is Betti SI S5.
# So set this, run N independent realizations, and average tau ACROSS runs - the
# exciton master equation sits outside the internal MC loop and cannot average itself.
DISORDER_FWHM_CM = float(__import__("os").environ.get("DISORDER_FWHM", "0"))
_DIS_SEED = int(__import__("os").environ.get("DISORDER_SEED", "-1"))
SITE_ENERGY_DIFF_CUTOFF = 300.0
V_CUTOFF = float(__import__('os').environ.get('V_CUTOFF', '60'))   # scan-able; 60 is the vibronic-convention default
R_C = float(__import__('os').environ.get('R_C_ANG', '1e-6'))   # v2: uncorrelated bath by default; set R_C_ANG=5.0 for the old correlated form
INTER_DOMAIN_DISTANCE_CUTOFF = 100.0

# Energies
DEFAULT_PSI_CORE_ENERGY = 14800.0
DEFAULT_FCPI_ENERGY = 14800.0

CORE_CHAIN_IDS = {'l','r','a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u', 'v', 'w', 'g'}
RC_IDS = [(_p.split(":")[0], int(_p.split(":")[1]))
          for _p in __import__("os").environ.get("RC_IDS", "a:803,b:804").split(",") if _p.strip()]
# Rates
import os as _oskc
CHARGE_SEP_RATE_PS = float(_oskc.environ.get('K_CS', '2.0'))
INTRINSIC_DISSIPATION_RATE_PS = float(_oskc.environ.get('K_DISS_PS', str(0.5 / 1000.0)))
MARKOV_TIME_STEP_PS = 0.2
SIMULATION_DURATION_PS = 4000.0
SIMULATION_STOP_THRESHOLD = 1e-5
PURE_DEPHASING_TIME_FS = 1e9

# ==============================================================================
#  3. SPECTRAL DENSITY & UTILS
# ==============================================================================

def spectral_density_raw(omega_radfs, S_total):
    w1 = RENGER_W1_CM / HBAR
    w2 = RENGER_W2_CM / HBAR
    s1 = RENGER_S1
    s2 = RENGER_S2

    w = np.atleast_1d(omega_radfs)
    J = np.zeros_like(w)
    mask = w > 1e-9
    w_safe = w[mask]

    term1 = (s1 / (factorial(7) * 2 * w1**4)) * w_safe**3 * np.exp(-np.sqrt(w_safe/w1))
    term2 = (s2 / (factorial(7) * 2 * w2**4)) * w_safe**3 * np.exp(-np.sqrt(w_safe/w2))
    J[mask] = (S_total / (s1 + s2)) * (term1 + term2)

    if np.ndim(omega_radfs) == 0: return J[0]
    return J

def bose_einstein(omega_radfs, temp_k):
    energy = HBAR * omega_radfs
    kT = KB * temp_k
    arg = energy / kT

    if np.ndim(arg) == 0:
        if arg > 50: return 0.0
        if arg < -50: return -1.0
        if np.abs(arg) < 1e-9: return 1.0/arg
        return 1.0 / (np.exp(arg) - 1.0)
    else:
        res = np.zeros_like(arg)
        mask_pos = arg > 1e-9
        mask_neg = arg < -1e-9
        res[mask_pos] = 1.0 / (np.exp(arg[mask_pos]) - 1.0)
        res[mask_neg] = 1.0 / (np.exp(arg[mask_neg]) - 1.0)
        mask_small = (~mask_pos) & (~mask_neg)
        res[mask_small] = 1.0 / (arg[mask_small] + 1e-12)
        return res


# --- Ratsep/Reimers intramolecular modes, FC SIDEBANDS ONLY (no -w_k t, no extra reorg shift) ---
import os as _ov
# --- phonon / intramolecular-mode boundary.  SETTLED at 300 cm-1 (2026-08-22). ------------
# Three independent anchors, all pointing at the same place:
#
#  ~90 cm-1   Ratsep, Pieper, Irrgang & Freiberg, JPCB 112, 110 (2008), CP29 dFLN:
#             "The broad feature present in the overlap region of phonon and vibrational
#              modes at about 90 cm-1 is characterized by S = 0.048."
#             Their difference spectrum "reveals a strong and broad low-frequency mode at
#              about 90 cm-1 as well as a weaker mode at 192 cm-1."
#             i.e. the experimentalists themselves cannot attribute the ~90 cm-1 intensity
#             to one component or the other.
#
#  ~260 cm-1  MEASURED HERE, this engine's own two components: the lambda density of the
#             69-mode table overtakes that of the B777 phonon between 250 and 300 cm-1
#             (modes/phonon = 0.14 on 200-250, 1.95 on 250-300), because the 263 cm-1 mode
#             arrives just as the phonon wing collapses.  Per 100 cm-1: 100-200 phonon x10.6,
#             200-300 phonon x1.1 (parity), 300-400 modes x3.8, 500-600 modes x11.9.
#
#  ~300 cm-1  Ratsep 2008 again:
#             "Up to frequencies of about 300 cm-1, the vibronic structure is superimposed
#              on the high-energy wing of the PSB, thus precluding an unambiguous separation
#              of phonon and vibrational features.  Therefore, a tentative separation is
#              attempted using the parameters of a simulated phonon wing."
#
# 300 assigns the phonon-dominated band to the phonon and leaves the mode-dominated band to
# the modes.  Going higher (400) would hand 300-400 to the phonon alone, where the modes
# outweigh it 3.8:1 and which holds the strongest mode below 700 cm-1 (349, S=0.0172) --
# the same error in the opposite direction.  Cost of 100 -> 300: +1.6% on tau.
_VIBWMIN = float(_ov.environ.get("VIB_WMIN", "300.0"))
VIB_MODES = []
if _ov.environ.get("VIBRONIC", "1") == "1":                 # SETTLED: modes required
    for _ln in open("chla_vibronic_TEA.txt"):
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#"):
            _w, _lam = float(_ln.split()[0]), float(_ln.split()[1])
            # v5: the phonon/mode boundary is now a stated knob, not a hard-coded 100.0.
            # Default 100 keeps the historical behaviour (drops the 22/66/99 cm-1 entries,
            # which sit inside the B777 phonon).  VIB_WMIN=0 includes the whole table.
            if _w >= _VIBWMIN: VIB_MODES.append((_w, _lam / _w))
    print("[VIBRONIC] %d modes above VIB_WMIN=%.0f cm-1, S_vib=%.4f, lambda_vib=%.1f cm-1 (sidebands only)"
          % (len(VIB_MODES), _VIBWMIN, sum(s for _, s in VIB_MODES), sum(s*w for w, s in VIB_MODES)))

# --- dipole convention, derived from whatever mode table was just loaded -------------
_SVIB_LOADED = sum(_s for _, _s in VIB_MODES)
MAGNITUDE_MU_A = sqrt(float(_ov.environ.get("MU2_A", repr(MU2_CHLA_00 * exp(_SVIB_LOADED)))))
TRESP_SCALE_AUTO = 0.8136 * (MAGNITUDE_MU_A / sqrt(MU2_CHLA_00))   # 0.8136 targets the 0-0
print("[DIPOLE] mu^2 = %.4f D^2 = %.1f * e^%.4f ; TrEsp charge scale = %.5f"
      % (MAGNITUDE_MU_A**2, MU2_CHLA_00, _SVIB_LOADED, TRESP_SCALE_AUTO))

def _add_vib(g_arr, t_axis, temp_k):
    for _wc, _S in VIB_MODES:
        _wk = _wc / HBAR
        _co = 1.0 / np.tanh((HBAR * _wk) / (2.0 * KB * temp_k))
        g_arr = g_arr + _S * (_co * (1.0 - np.cos(_wk * t_axis)) + 1j * np.sin(_wk * t_axis))
    return g_arr

def precalculate_g_function(t_axis_fs, s_factor, temp_k):
    w_axis_cm = np.arange(FREQ_STEP_CM, FREQ_MAX_CM, FREQ_STEP_CM)
    w_axis_radfs = w_axis_cm / HBAR
    J_val = spectral_density_raw(w_axis_radfs, s_factor)
    beta_term = (HBAR * w_axis_radfs) / (2 * KB * temp_k)
    coth_term = 1.0 / np.tanh(beta_term)
    base = J_val

    g_t_array = np.zeros(len(t_axis_fs), dtype=complex)
    for i, t in enumerate(t_axis_fs):
        wt = w_axis_radfs * t
        integ_real = base * coth_term * (1.0 - np.cos(wt))
        g_real = simpson(y=integ_real, x=w_axis_radfs)
        integ_imag = base * (np.sin(wt))
        g_imag = simpson(y=integ_imag, x=w_axis_radfs)
        g_t_array[i] = g_real + 1j * g_imag
    if VIB_MODES: g_t_array = _add_vib(g_t_array, t_axis_fs, temp_k)
    return g_t_array

def precalculate_lambda(s_factor):
    w_axis_cm = np.arange(FREQ_STEP_CM, FREQ_MAX_CM, FREQ_STEP_CM)
    w_axis_radfs = w_axis_cm / HBAR
    J_val = spectral_density_raw(w_axis_radfs, s_factor)
    integrand = w_axis_radfs * J_val
    lambda_radfs = simpson(y=integrand, x=w_axis_radfs)
    return lambda_radfs * HBAR

# ==============================================================================
#  4. PHYSICS MODULES
# ==============================================================================

# --- P700-resolved Huang-Rhys factor (S=1 on the P700 Chls only) ---
_LAMBDA_UNIT = [None]
def _lambda_unit():
    if _LAMBDA_UNIT[0] is None:
        _LAMBDA_UNIT[0] = precalculate_lambda(1.0)
    return _LAMBDA_UNIT[0]

def _pigment_S(chl):
    """Per-pigment phonon Huang-Rhys used by the EXCITON-level rates."""
    if chl.get('is_rc', False):
        return HUANG_RHYS_S_RC
    if HUANG_RHYS_S_REDX is not None and chl.get('is_red', False):
        return HUANG_RHYS_S_REDX
    return HUANG_RHYS_S_ANT

def _gamma_S(domain, chlorophylls):
    '''Sum_i |c_ia|^2 |c_ib|^2 S_i. Because J is linear in S, this equals pyQME's
    per-spectral-density-group decomposition Sum_Z gamma_ab^Z J(w,S_Z) / J(w,1) exactly.
    Valid in the uncorrelated-bath limit R_C -> 0 (the v2 default, R_C_ANG=1e-6).'''
    C2 = np.asarray(domain['local_coefficients'])**2
    Sm = np.array([_pigment_S(chlorophylls[g]) for g in domain['global_indices']])
    return np.einsum('ia,ib,i->ab', C2, C2, Sm)

def _S_per_exciton(domain, chlorophylls):
    C = domain['local_coefficients']
    Sm = np.array([_pigment_S(chlorophylls[g]) for g in domain['global_indices']])
    return (C**2 * Sm[:, None]).sum(axis=0)

def calculate_redfield_rate_element(omega_ab_cm, gamma_ab, s_factor, temp_k):
    omega_radfs = omega_ab_cm / HBAR
    if abs(omega_radfs) < 1e-9: return 0.0

    if omega_radfs > 0:
        J_val = spectral_density_raw(omega_radfs, s_factor)
        n_val = bose_einstein(omega_radfs, temp_k)
        rate = 2.0 * np.pi * gamma_ab * (omega_radfs**2) * J_val * (1.0 + n_val)
    else:
        w_pos = -omega_radfs
        J_val = spectral_density_raw(w_pos, s_factor)
        n_val = bose_einstein(w_pos, temp_k)
        rate = 2.0 * np.pi * gamma_ab * (w_pos**2) * J_val * n_val

    return rate * 1000.0

def calculate_redfield_rates(domain, chlorophylls, temp_k=TEMP_K, energies=None):
    num_excitons = len(domain['local_energies'])
    rates = np.zeros((num_excitons, num_excitons))
    gamma = calculate_gamma(domain, chlorophylls)

    # --- MODIFIED: CHECK FOR RED FORMS ---
    # We check if *any* pigment in this domain is flagged as red
    is_red_domain = any(chlorophylls[g_idx]['is_red'] for g_idx in domain['global_indices'])
    is_p700_domain = any(chlorophylls[g_idx]['is_rc'] for g_idx in domain['global_indices'])
    S_ex = _S_per_exciton(domain, chlorophylls)
    _spg = os.environ.get('S_PER_GROUP', '1') == '1'
    _gS = _gamma_S(domain, chlorophylls) if _spg else None
    # -------------------------------------

    for alpha in range(num_excitons):
        for beta in range(num_excitons):
            if alpha == beta: continue
            _EE = domain['local_energies'] if energies is None else energies
            omega_ab_cm = _EE[alpha] - _EE[beta]
            rates[alpha, beta] = calculate_redfield_rate_element(
                omega_ab_cm,
                (_gS[alpha, beta] if _spg else gamma[alpha, beta]),
                (1.0 if _spg else 0.5*(S_ex[alpha]+S_ex[beta])), temp_k
            )
    return rates

def calculate_renormalized_energies_full(domain, chlorophylls, lambda_core, lambda_ant, lambda_red):
    num_excitons = len(domain['local_energies'])
    gamma = calculate_gamma(domain, chlorophylls)
    energies_cm = domain['local_energies']

    # --- Check Red/Core status (Same as before) ---
    is_red_domain = any(chlorophylls[g_idx]['is_red'] for g_idx in domain['global_indices'])
    is_p700_domain = any(chlorophylls[g_idx]['is_rc'] for g_idx in domain['global_indices'])
    S_ex = _S_per_exciton(domain, chlorophylls)
    _spg = os.environ.get('S_PER_GROUP', '1') == '1'
    _gS = _gamma_S(domain, chlorophylls) if _spg else None
    _S_CUR = [S_ex[0] if len(S_ex) else HUANG_RHYS_S_ANT]
    # -------------------------------------

    renormalized_energies = np.zeros(num_excitons)
    W_MIN = 1.0 / HBAR
    W_MAX = float(os.environ.get('PV_WMAX_CM', '2000.0')) / HBAR

    def integrand_numerator(w_radfs):
        if w_radfs > 0:
            j_w = spectral_density_raw(w_radfs, _S_CUR[0])
            n_w = bose_einstein(w_radfs, TEMP_K)
            return (w_radfs**2) * (1.0 + n_w) * j_w
        if w_radfs < 0:
            # Akhtar Eq.15 second term: n(-w) J(-w), i.e. the omega<0 branch
            j_w = spectral_density_raw(-w_radfs, _S_CUR[0])
            n_w = bose_einstein(-w_radfs, TEMP_K)
            return (w_radfs**2) * n_w * j_w
        return 0.0

    _pv_eps = float(os.environ.get('PV_EPSREL', '1e-2'))
    _pv_lim = int(os.environ.get('PV_LIMIT', '20'))
    _pv_cut = float(os.environ.get('PV_GAMMA_CUT', '1e-4'))
    _pv_full = os.environ.get('PV_FULL_RANGE', '1') == '1'   # omega<0 branch of Akhtar Eq.15; default ON, set 0 only to reproduce pre-2026-08-22 numbers
    W_MAX_L = float(os.environ.get('PV_WMAX_CM', '2000.0')) / HBAR
    _reorg_here = np.zeros(num_excitons); _disp_here = np.zeros(num_excitons)
    for alpha in range(num_excitons):
        _S_CUR[0] = 1.0 if _spg else S_ex[alpha]
        reorg_shift = -(_gS[alpha, alpha] if _spg else gamma[alpha, alpha] * S_ex[alpha]) * _lambda_unit()
        dispersive_shift = 0.0
        for beta in range(num_excitons):
            if alpha == beta: continue

            w_ab_radfs = (energies_cm[alpha] - energies_cm[beta]) / HBAR
            gamma_ab = _gS[alpha, beta] if _spg else gamma[alpha, beta]
            
            # OPTIMIZATION: Check if coupling is significant before integrating
            if abs(gamma_ab) < _pv_cut: continue

            def _pv_over(_a, _b):
                if _a < w_ab_radfs < _b:
                    _v, _ = quad(integrand_numerator, _a, _b,
                                 weight='cauchy', wvar=w_ab_radfs,
                                 limit=_pv_lim, epsrel=_pv_eps)
                    return -_v
                def _ff(w):
                    _d = w_ab_radfs - w
                    if abs(_d) < 1e-12: return 0.0
                    return integrand_numerator(w) / _d
                _v, _ = quad(_ff, _a, _b, limit=_pv_lim, epsrel=_pv_eps)
                return _v
            val = _pv_over(W_MIN, W_MAX)
            if _pv_full:
                val += _pv_over(-W_MAX, -W_MIN)

            dispersive_shift += gamma_ab * val

        total_shift = reorg_shift + (dispersive_shift * HBAR)
        _reorg_here[alpha] = reorg_shift; _disp_here[alpha] = dispersive_shift * HBAR
        renormalized_energies[alpha] = energies_cm[alpha] + total_shift

    _REORG_PARTS.append(_reorg_here); _DISP_PARTS.append(_disp_here)
    return renormalized_energies

# DEAD in the v2/v5 path: the grid-based lineshape (13000-16000 cm-1) from v1.  Not called.
# NOTE it hard-codes the lifetime broadening (0.5*gamma_life) with no flag - do not revive
# it without checking that against LINESHAPE_GAMMA.
def calculate_optical_lineshape_full(omega_range_cm, E_0_cm, gamma_aa, lifetime_ps, is_emission, precalculated_g_t, t_axis_fs):
    lineshape = np.zeros_like(omega_range_cm)
    gamma_life = 1.0 / (lifetime_ps * 1000.0)
    gamma_pure = 1.0 / PURE_DEPHASING_TIME_FS
    real_decay_rate = ( 0.5*gamma_life) + gamma_pure
    decay = np.exp(-real_decay_rate * t_axis_fs)
    g_t_scaled = gamma_aa * precalculated_g_t
    G_term = np.exp(-(g_t_scaled - g_t_scaled[0]))

    w_range_radfs = omega_range_cm / HBAR
    w_0_radfs = E_0_cm / HBAR

    for i, w in enumerate(w_range_radfs):
        dw = w - w_0_radfs
        if is_emission:
            phase = np.exp(-1j * dw * t_axis_fs)
        else:
            phase = np.exp(1j * dw * t_axis_fs)

        integrand = phase * G_term * decay
        res = simpson(y=integrand, x=t_axis_fs)
        lineshape[i] = np.real(res)

    area = simpson(y=lineshape, x=omega_range_cm)
    if area > 1e-15:
        return lineshape / area
    else:
        return np.zeros_like(lineshape)

# DEAD in the v2/v5 path: the v1 grid-based generalised Forster overlap.  Not called; the
# live routine is calculate_gf_rates_timedomain below.  Kept only for reference.
def calculate_generalized_forster_rates(V_eff, emission_shapes1, absorption_shapes2, omega_grid_cm):
    num_excitons1, num_excitons2 = V_eff.shape
    rates = np.zeros((num_excitons1, num_excitons2))
    prefactor = 2 * np.pi / HBAR

    for alpha in range(num_excitons1):
        ems = emission_shapes1[alpha]
        if np.sum(ems) == 0: continue

        for beta in range(num_excitons2):
            abs_ = absorption_shapes2[beta]
            if np.sum(abs_) == 0: continue

            coupling = V_eff[alpha, beta]
            overlap = simpson(y=(ems * abs_), x=omega_grid_cm)
            rate_fs = prefactor * (coupling**2) * overlap
            final_rate = rate_fs * 1000.0
            if final_rate > 5000.0: final_rate = 5000.0
            rates[alpha, beta] = final_rate
    return rates

def calculate_gf_rates_timedomain(V_eff, D, A, t_axis_fs):
    """Grid-free generalised Forster: the exact Fourier transform of int dw F_D A_A
       built from the SAME lineshape functions.  Both ghat unconjugated
       (= pyQME tensors/markov/forster.py).  Detailed balance exact by construction."""
    nD, nA = V_eff.shape
    rates = np.zeros((nD, nA))
    for a in range(nD):
        gD = D["g"][a]; eD = D["E"][a]; GD = D["deph"][a]
        for b in range(nA):
            V = V_eff[a, b]
            if abs(V) < 1e-10: continue
            gA = A["g"][b]; eA = A["E"][b]; GA = A["deph"][b]
            integ = np.exp(1j * (eD - eA) / HBAR * t_axis_fs - gD - gA - (GD + GA) * t_axis_fs)
            val = float(np.real(simpson(y=integ, x=t_axis_fs)))
            r = 2.0 * ((V / HBAR) ** 2) * val * 1000.0
            _CLAMP['n'] += 1
            if r < 0.0:
                _CLAMP['neg'] += 1
                if r < _CLAMP['negmin']: _CLAMP['negmin'] = r
                r = 0.0
            if r > 5000.0:
                _CLAMP['hi'] += 1
                r = 5000.0
            rates[a, b] = r
    return rates


def calculate_weighted_trapping_rates(domain_results, chlorophylls, intrinsic_rate_ps, temp_k):
    num_domains = len(domain_results)
    effective_rates = np.zeros(num_domains)

    for i, domain in enumerate(domain_results):
        energies = domain['local_energies']
        coeffs = domain['local_coefficients']

        rc_local_indices = []
        for local_idx, global_idx in enumerate(domain['global_indices']):
            chl = chlorophylls[global_idx]
            if (chl['chain_id'], chl['res_seq']) in RC_IDS:
                rc_local_indices.append(local_idx)

        if not rc_local_indices:
            effective_rates[i] = 0.0
            continue

        num_excitons = len(energies)
        k_alpha = np.zeros(num_excitons)

        _use_part = _oskc.environ.get('USE_PARTICIPATION', '0') == '1'
        for alpha in range(num_excitons):
            if _use_part:
                participation = float(np.sum(np.asarray(coeffs)[rc_local_indices, alpha] ** 2))
            else:
                participation = 1.0
            k_alpha[alpha] = intrinsic_rate_ps * participation

        min_E = np.min(energies)
        boltzmann = np.exp(-(energies - min_E) / (KB * temp_k))
        Z = np.sum(boltzmann)

        if Z > 1e-12:
            effective_rates[i] = np.sum(k_alpha * boltzmann) / Z
        else:
            effective_rates[i] = 0.0
        print("  RC-SINK domain %d: n_chl=%d n_rc=%d  k_eff=%.4f ps^-1  (tau_eff=%.2f ps)"
              % (i, len(domain['global_indices']), len(rc_local_indices), effective_rates[i],
                 (1.0 / effective_rates[i]) if effective_rates[i] > 0 else float('inf')))
    return effective_rates

# ==============================================================================
#  5. HELPER FUNCTIONS
# ==============================================================================

def read_pdb_data(filename):
    chls_dict = {}
    try:
        with open(filename, 'r') as pdb_file:
            for line in pdb_file:
                res_name_check = line[17:20].strip()
                if (line.startswith("HETATM") and res_name_check in {"CLA","CL0","CHL","CLB","KC1","KC2"}):
                    res_name, chain_id = line[17:20].strip(), line[21].strip()
                    res_seq = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    chl_key = (chain_id, res_seq)
                    if chl_key not in chls_dict:
                        is_rc = chl_key in RC_IDS
                        # Initialize 'is_red' as False, we will update it later
                        chls_dict[chl_key] = {"chain_id": chain_id, "res_seq": res_seq,
                                              "all_atoms": [], "mg_coords": None,
                                              "nb_coords": None, "nd_coords": None,
                                              "is_rc": is_rc, "is_red": False, "atom_by_name": {}, "ptype": ('b' if res_name in {"CHL","CLB"} else ('c' if res_name in {"KC1","KC2"} else 'a'))}
                    chls_dict[chl_key]["all_atoms"].append(np.array([x, y, z]))
                    atom_name = line[12:16].strip()
                    chls_dict[chl_key]["atom_by_name"].setdefault(atom_name, np.array([x, y, z]))
                    if atom_name == 'MG': chls_dict[chl_key]["mg_coords"] = np.array([x, y, z])
                    elif atom_name == 'NB': chls_dict[chl_key]["nb_coords"] = np.array([x, y, z])
                    elif atom_name == 'ND': chls_dict[chl_key]["nd_coords"] = np.array([x, y, z])
    except FileNotFoundError:
        print(f"Error: PDB file '{filename}' not found."); exit()
    return list(chls_dict.values())

# DEAD: the geometric red-form rule was replaced by assignment from site_energies.txt in
# __main__.  Not called.  RED_FORM_DISTANCE_CUTOFF is therefore inert.
def read_cif_data(filename):
    """mmCIF (PDBx) reader for the same fields read_pdb_data extracts.  Column order is
    the standard _atom_site loop: 3 label_atom_id, 5 label_comp_id, 10-12 Cartn_x/y/z,
    16 auth_seq_id, 18 auth_asym_id."""
    chls_dict = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                parts = line.split()
                if len(parts) < 19:
                    continue
                res_name = parts[5]
                if res_name not in {"CLA", "CL0", "CHL", "CLB", "KC1", "KC2"}:
                    continue
                atom_name = parts[3].strip('"')
                chain_id = parts[18]
                try:
                    res_seq = int(parts[16])
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                except ValueError:
                    continue
                chl_key = (chain_id, res_seq)
                if chl_key not in chls_dict:
                    chls_dict[chl_key] = {"chain_id": chain_id, "res_seq": res_seq,
                                          "all_atoms": [], "mg_coords": None,
                                          "nb_coords": None, "nd_coords": None,
                                          "is_rc": chl_key in RC_IDS, "is_red": False,
                                          "atom_by_name": {},
                                          "ptype": ('b' if res_name in {"CHL", "CLB"} else
                                                    ('c' if res_name in {"KC1", "KC2"} else 'a'))}
                xyz = np.array([x, y, z])
                chls_dict[chl_key]["all_atoms"].append(xyz)
                chls_dict[chl_key]["atom_by_name"].setdefault(atom_name, xyz)
                if atom_name == "MG":   chls_dict[chl_key]["mg_coords"] = xyz
                elif atom_name == "NB": chls_dict[chl_key]["nb_coords"] = xyz
                elif atom_name == "ND": chls_dict[chl_key]["nd_coords"] = xyz
    except FileNotFoundError:
        print("Error: structure file '%s' not found." % filename); exit()
    return list(chls_dict.values())


def read_structure(filename):
    """dispatch on extension so one engine serves .pdb and .cif species"""
    if filename.lower().endswith((".cif", ".mmcif")):
        return read_cif_data(filename)
    return read_pdb_data(filename)


def identify_red_forms(chlorophylls):
    """
    Identifies red chlorophylls based on Mg-Mg distances < 10.0 Angstroms.
    Updates the 'is_red' flag in the chlorophyll dictionary.
    Returns a set of red chlorophyll keys.
    """
    red_indices = set()
    print("\n--- Identifying Potential Red Forms (Dimers < {:.1f} A) ---".format(RED_FORM_DISTANCE_CUTOFF))

    n_chls = len(chlorophylls)
    for i in range(n_chls):
        for j in range(i + 1, n_chls):
            center_i = chlorophylls[i]["mg_coords"] if chlorophylls[i]["mg_coords"] is not None else chlorophylls[i]["center"]
            center_j = chlorophylls[j]["mg_coords"] if chlorophylls[j]["mg_coords"] is not None else chlorophylls[j]["center"]

            dist = np.linalg.norm(center_i - center_j)

            if dist < RED_FORM_DISTANCE_CUTOFF:
                # We exclude P700 (Special Pair) from being treated as 'Red' antenna
                if chlorophylls[i]['is_rc'] or chlorophylls[j]['is_rc']:
                    continue

                chlorophylls[i]['is_red'] = True
                chlorophylls[j]['is_red'] = True
                red_indices.add(i)
                red_indices.add(j)
                print(f"  RED PAIR FOUND: {chlorophylls[i]['chain_id']}:{chlorophylls[i]['res_seq']} - "
                      f"{chlorophylls[j]['chain_id']}:{chlorophylls[j]['res_seq']} (Dist: {dist:.2f} A)")

    if not red_indices:
        print("  No red forms found based on distance criteria.")

    return chlorophylls

def read_site_energies(filename):
    site_energies = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3: site_energies[(parts[0], int(parts[1]))] = float(parts[2])
    except FileNotFoundError: print(f"Warning: '{filename}' not found. Using defaults.")
    return site_energies

def assign_site_energies(chlorophylls, site_energies):
    for chl in chlorophylls:
        key = (chl["chain_id"], chl["res_seq"])
        if site_energies and key in site_energies:
            chl["site_energy"] = site_energies[key]
        elif chl.get("ptype")=="b":
            chl["site_energy"]=float(__import__("os").environ.get("CHLB_E","15400"))
        elif chl.get("ptype")=="c":
            chl["site_energy"]=float(__import__("os").environ.get("CHLC_E","15786"))
        else:
            chl["site_energy"] = DEFAULT_PSI_CORE_ENERGY if chl["chain_id"] in CORE_CHAIN_IDS else DEFAULT_FCPI_ENERGY
    return chlorophylls

def calculate_centers_and_dipoles(chlorophylls):
    for chl in chlorophylls:
        chl["center"] = chl["mg_coords"] if chl["mg_coords"] is not None else np.mean(chl["all_atoms"], axis=0)
        if chl["nb_coords"] is not None and chl["nd_coords"] is not None:
            vec = chl["nd_coords"] - chl["nb_coords"]
            norm = np.linalg.norm(vec)
            _mag={"a":MAGNITUDE_MU_A,"b":sqrt(20.33),"c":sqrt(float(__import__("os").environ.get("CHLC_DIP2", repr(0.400787*MAGNITUDE_MU_A**2))))}.get(chl.get("ptype","a"),MAGNITUDE_MU_A); chl["dipole_vec"] = _mag * (vec / norm) if norm > 1e-6 else np.zeros(3)
        else: chl["dipole_vec"] = np.zeros(3)
    return chlorophylls

# === TrEsp (Chl a only); Madjet 2006 B3LYP charges; scalar = 116141*C_FACTOR ===
USE_TRESP = os.environ.get("USE_TRESP", "1") == "1"         # SETTLED
PREFACTOR_TRESP = 116141.0 * C_FACTOR
_AC_SCALE = float(__import__("os").environ.get("CHLC_AC_SCALE", "1.0"))
TRESP_SCALE = float(__import__('os').environ.get('TRESP_SCALE', repr(TRESP_SCALE_AUTO)))
CHG_CHLA_RAW = {
 "MG":-0.0216740,"CHA":0.1067790,"CHB":-0.0486960,"CHC":-0.0987250,"CHD":0.0727260,
 "NA":0.0316830,"C1A":-0.1308200,"C2A":0.0100480,"C3A":0.0023890,"C4A":0.0779830,
 "CMA":0.0055560,"CAA":-0.0010500,"CBA":0.0007350,"CGA":-0.0080400,"O1A":-0.0013300,
 "O2A":0.0070910,"NB":-0.0622970,"C1B":0.0811220,"C2B":0.0047770,"C3B":-0.0092040,
 "C4B":0.1062710,"CMB":0.0169630,"CAB":0.0106810,"CBB":0.0349470,"NC":-0.0121660,
 "C1C":0.0836460,"C2C":-0.0074200,"C3C":-0.0011260,"C4C":-0.0440080,"CMC":-0.0051610,
 "CAC":0.0080200,"CBC":0.0010120,"ND":0.1082920,"C1D":-0.1108120,"C2D":-0.0119810,
 "C3D":0.0087990,"C4D":-0.1250440,"CMD":-0.0251560,"CAD":-0.0192500,"OBD":-0.0200440,
 "CBD":-0.0112380,"CGD":0.0059790,"O1D":-0.0053880,"O2D":0.0018110,"CED":-0.0052560,
 "C1":-0.0014240,
}
# --- Chl c per-pigment TrEsp: superpose CHELPG template onto each Chl c core ---
try:
    import json as _json
    _CT = _json.load(open("chlc_template.json"))
    _CHLC_COORDS = np.array(_CT["coords"]); _CHLC_Q = np.array(_CT["charges"])
    _CHLC_CORE = np.array([_CT["core"][k] for k in ("MG", "NA", "NB", "NC", "ND")])
    CHLC_SCALE = float(os.environ.get("CHLC_TARGET_D", repr(sqrt(0.400787)*MAGNITUDE_MU_A))) / _CT["mu_D"]   # Chl c dipole = empirical Chl a x QC ratio (~1.68 D)
except Exception as _e:
    _CHLC_COORDS = None
def _kabsch_c(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)
def _setup_chlc_tresp(chlorophylls):
    if not USE_TRESP or _CHLC_COORDS is None:
        return chlorophylls
    n = 0
    for chl in chlorophylls:
        if chl.get("ptype", "a") != "c":
            continue
        abn = chl.get("atom_by_name", {})
        if not all(k in abn for k in ("MG", "NA", "NB", "NC", "ND")):
            continue  # missing core -> leave empty -> point-dipole fallback
        tgt = np.array([abn[k] for k in ("MG", "NA", "NB", "NC", "ND")])
        R, t = _kabsch_c(_CHLC_CORE, tgt)
        q = _CHLC_Q * CHLC_SCALE
        q = q - q.mean()
        chl["tresp_q"] = q
        chl["tresp_xyz"] = (R @ _CHLC_COORDS.T).T + t
        n += 1
    print("[TrEsp] Chl c clouds (superposed): %d" % n)
    return chlorophylls

def build_tresp_clouds(chlorophylls):
    n_a = n_full = 0
    for chl in chlorophylls:
        if chl.get("ptype", "a") != "a":
            chl["tresp_q"] = np.zeros(0); chl["tresp_xyz"] = np.zeros((0, 3)); continue
        n_a += 1
        abn = chl.get("atom_by_name", {})
        names = [a for a in CHG_CHLA_RAW if a in abn]
        q = np.array([CHG_CHLA_RAW[a] * TRESP_SCALE for a in names])
        if q.size > 0: q = q - q.sum() / q.size
        chl["tresp_q"] = q
        chl["tresp_xyz"] = np.array([abn[a] for a in names]) if names else np.zeros((0, 3))
        if q.size == len(CHG_CHLA_RAW): n_full += 1
    if USE_TRESP:
        print(f"[TrEsp] Chl a clouds: {n_a} pigments ({n_full} complete); Chl b/c point-dipole; scale={TRESP_SCALE}")
    return chlorophylls
def tresp_coupling(chl_m, chl_n):
    qm, Xm = chl_m["tresp_q"], chl_m["tresp_xyz"]
    qn, Xn = chl_n["tresp_q"], chl_n["tresp_xyz"]
    if qm.size == 0 or qn.size == 0: return None
    diff = Xm[:, None, :] - Xn[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    return PREFACTOR_TRESP * float((qm[:, None] * qn[None, :] / dist).sum())

def calculate_hamiltonian(chlorophylls, stochastic=False):
    num_chls = len(chlorophylls)
    v_mn, h_ex = np.zeros((num_chls, num_chls)), np.zeros((num_chls, num_chls))
    disorder_std = DISORDER_FWHM_CM / 2.35482
    for m in range(num_chls):
        h_ex[m, m] = chlorophylls[m]["site_energy"] + (np.random.normal(0, disorder_std) if stochastic else 0)
        for n in range(m + 1, num_chls):
            r_vec = chlorophylls[n]["center"] - chlorophylls[m]["center"]
            R = np.linalg.norm(r_vec)
            if R > 1.0:
                v_tresp = tresp_coupling(chlorophylls[m], chlorophylls[n]) if USE_TRESP else None
                if v_tresp is not None:
                    v_temp = v_tresp
                else:
                    r_vec_nm, R_nm = r_vec / 10.0, R / 10.0
                    mu_m, mu_n = chlorophylls[m]["dipole_vec"], chlorophylls[n]["dipole_vec"]
                    term1 = np.dot(mu_m, mu_n) / (R_nm**3)
                    term2 = 3.0 * np.dot(mu_m, r_vec_nm) * np.dot(mu_n, r_vec_nm) / (R_nm**5)
                    v_temp = VMN_PREFACTOR * C_FACTOR * (term1 - term2)
                if _AC_SCALE != 1.0 and {chlorophylls[m].get("ptype", "a"), chlorophylls[n].get("ptype", "a")} == {"a", "c"}:
                    v_temp *= _AC_SCALE
                if v_temp > 5000.0: v_temp = 5000.0
                if v_temp < -5000.0: v_temp = -5000.0
                v_mn[m, n] = v_mn[n, m] = h_ex[m, n] = h_ex[n, m] = v_temp
            else: v_mn[m, n] = v_mn[n, m] = 0.0
    return v_mn, h_ex


def _pr_domains(v_mn, h_ex, chlorophylls):
    """Betti & Cupellini SI S5: domains from the disorder-averaged participation-ratio matrix."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    fwhm = float(os.environ.get("PR_FWHM", "180"))
    nreal = int(os.environ.get("PR_NREAL", "400"))
    cut = float(os.environ.get("PR_CUT", "0.8"))
    seed = int(os.environ.get("PR_SEED", "70000"))
    N = h_ex.shape[0]
    E0 = np.diag(h_ex).copy()
    is_c = np.array([c.get("ptype", "a") == "c" for c in chlorophylls])
    H0 = h_ex.copy()
    mix = np.logical_xor(is_c[:, None], is_c[None, :])
    H0[mix] = 0.0
    np.fill_diagonal(H0, E0)
    rng = np.random.default_rng(seed)
    sigma = fwhm / 2.35482
    PR = np.zeros((N, N))
    for _ in range(nreal):
        H = H0.copy()
        np.fill_diagonal(H, E0 + rng.normal(0.0, sigma, N))
        _, C = np.linalg.eigh(H)
        P = C ** 2
        PR += P @ P.T
    PR /= nreal
    d = 1.0 - PR / np.sqrt(np.outer(np.diag(PR), np.diag(PR)))
    d = np.clip((d + d.T) / 2.0, 0.0, None)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(Z, t=cut, criterion="distance")
    doms = {}
    for i, l in enumerate(lab):
        doms.setdefault(l, []).append(i)
    out = list(doms.values())
    sz = sorted((len(x) for x in out), reverse=True)
    print("[PR-DOMAINS] FWHM=%.0f nreal=%d cut=%.2f -> n_domains=%d largest=%d top5=%s monomers=%d"
          % (fwhm, nreal, cut, len(out), sz[0], sz[:5], sum(1 for x in sz if x == 1)))
    return out


def partition_into_exciton_domains(v_mn, h_ex, v_cutoff, e_cutoff):
    site_energies = np.diag(h_ex)
    adj = (np.abs(v_mn) >= v_cutoff) & (np.abs(site_energies[:, np.newaxis] - site_energies) <= e_cutoff)
    np.fill_diagonal(adj, 0)
    _, labels = connected_components(csgraph=csr_matrix(adj), directed=False, return_labels=True)
    domain_map = defaultdict(list)
    for i, label in enumerate(labels): domain_map[label].append(i)
    return list(domain_map.values())

def get_stochastic_domain_results(h_ex_global, domains, chlorophylls):
    domain_results = []
    for i, domain_indices in enumerate(domains):
        domain_block = h_ex_global[np.ix_(domain_indices, domain_indices)]
        E_local, C_local = linalg.eigh(domain_block)
        domain_results.append({
            'global_indices': domain_indices,
            'local_energies': E_local,
            'local_coefficients': C_local,
            'chlorophylls': [(chlorophylls[idx]['chain_id'], chlorophylls[idx]['res_seq']) for idx in domain_indices]
        })
    return domain_results

def calculate_gamma(domain, chlorophylls):
    num_excitons = len(domain['local_energies'])
    gamma = np.zeros((num_excitons, num_excitons))
    
    # Pre-fetch centers to avoid dictionary lookups in the loop
    centers = [chlorophylls[g_idx]['center'] for g_idx in domain['global_indices']]
    coeffs = domain['local_coefficients']
    
    # Optimization: Pre-calculate distances and validity mask
    # If pigments are > 4 * R_C apart, exp(-dist/Rc) is < 0.018 (negligible)
    num_pigments = len(centers)
    valid_pairs = []
    
    SPATIAL_CUTOFF = 4.0 * R_C 
    
    for m in range(num_pigments):
        for n in range(num_pigments):
            dist = np.linalg.norm(centers[m] - centers[n])
            if dist < SPATIAL_CUTOFF:
                valid_pairs.append((m, n, np.exp(-dist / R_C)))

    # COEFF_CUTOFF: If an exciton has < 1% density on a pigment, ignore it for rates
    COEFF_CUTOFF = 0.01 

    for alpha in range(num_excitons):
        for beta in range(num_excitons):
            sum_gamma = 0.0
            
            # Use the pre-filtered sparse list of valid geometric pairs
            for m, n, corr_factor in valid_pairs:
                # Get coefficients
                c_ma = coeffs[m, alpha]
                c_na = coeffs[n, alpha]
                c_mb = coeffs[m, beta]
                c_nb = coeffs[n, beta]

                # Optimization: Skip if exciton participation is tiny
                prod = c_ma * c_na * c_mb * c_nb
                if abs(prod) < 1e-10: 
                    continue
                
                sum_gamma += prod * corr_factor
                
            gamma[alpha, beta] = sum_gamma
            
    return gamma

def calculate_interdomain_coupling(domain1, domain2, v_mn_global):
    num_excitons1, num_excitons2 = len(domain1['local_energies']), len(domain2['local_energies'])
    V_eff = np.zeros((num_excitons1, num_excitons2))
    for alpha in range(num_excitons1):
        for beta in range(num_excitons2):
            sum_v = 0.0
            for m_idx, m_global in enumerate(domain1['global_indices']):
                for n_idx, n_global in enumerate(domain2['global_indices']):
                    sum_v += domain1['local_coefficients'][m_idx, alpha] * domain2['local_coefficients'][n_idx, beta] * \
                             v_mn_global[m_global, n_global]
            V_eff[alpha, beta] = sum_v
    return V_eff

def get_min_domain_distance(domain1, domain2, all_chlorophylls):
    min_dist_sq = np.inf
    centers1 = [all_chlorophylls[i]['center'] for i in domain1['global_indices']]
    centers2 = [all_chlorophylls[i]['center'] for i in domain2['global_indices']]
    for c1 in centers1:
        for c2 in centers2:
            dist_sq = np.sum((c1 - c2)**2)
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq
    return np.sqrt(min_dist_sq)

def load_domain_info(exciton_domains_indices, chlorophylls):
    domain_info = []
    rc_indices = set()
    _, h_ex_mean = calculate_hamiltonian(chlorophylls, stochastic=False)
    for i, domain_indices in enumerate(exciton_domains_indices):
        domain_chls = []; is_rc_domain = False; _mu2 = 0.0
        for chl_idx in domain_indices:
            chl = chlorophylls[chl_idx]
            chl_id_tuple = (chl['chain_id'], chl['res_seq'])
            domain_chls.append(chl_id_tuple)
            _dv = chl.get('dipole_vec')
            _mu2 += float(np.dot(_dv, _dv)) if _dv is not None else 0.0
            if chl_id_tuple in RC_IDS: is_rc_domain = True
        if is_rc_domain: rc_indices.add(i)
        domain_block = h_ex_mean[np.ix_(domain_indices, domain_indices)]
        E_local_mean, _ = linalg.eigh(domain_block)
        domain_info.append({'chlorophylls': domain_chls, 'energies': E_local_mean, 'size': len(domain_indices), 'mu2': _mu2})
    return domain_info, sorted(list(rc_indices))

def calculate_domain_level_rates(current_domain_results, inter_exciton_rates, rc_indices):
    num_domains = len(current_domain_results)
    domain_rates = np.zeros((num_domains, num_domains))
    for (d1, d2), exciton_rates in inter_exciton_rates.items():
        energies1 = np.array(current_domain_results[d1]['local_energies'])
        if len(energies1) > 0:
            min_energy1 = np.min(energies1)
            boltzmann1 = np.exp(-(energies1 - min_energy1) / (KB * TEMP_K))
            Z = np.sum(boltzmann1)
            if Z > 1e-9:
                thermal_probs1 = boltzmann1 / Z
                domain_rates[d2, d1] = np.sum(thermal_probs1[:, np.newaxis] * exciton_rates)
    return domain_rates

def run_simulation(avg_domain_rates, trapping_rates_vector, domain_info):
    num_domains = len(domain_info)
    Q = np.zeros((num_domains + 2, num_domains + 2))
    CHARGE_SEP_IDX, DISSIPATION_IDX = num_domains, num_domains + 1

    # MATRIX CLAMPING
    avg_domain_rates = np.clip(avg_domain_rates, 0.0, 500.0)

    Q[:num_domains, :num_domains] = avg_domain_rates
    for i in range(num_domains):
        Q[DISSIPATION_IDX, i] = INTRINSIC_DISSIPATION_RATE_PS
        # Weighted rate for this domain
        Q[CHARGE_SEP_IDX, i] = trapping_rates_vector[i]
        Q[i, i] = -np.sum(Q[:, i])

    # Nan/Inf protection
    if not np.all(np.isfinite(Q)):
        Q = np.nan_to_num(Q, nan=0.0, posinf=500.0, neginf=0.0)

    P = expm(Q * MARKOV_TIME_STEP_PS)

    x0 = np.zeros(num_domains + 2)
    valid_indices = []
    total_valid_pigments = 0
    for i, dom in enumerate(domain_info):
        match_count = sum(1 for chl in dom['chlorophylls'] if chl[0] in INITIAL_EXCITATION_CHAINS)
        if match_count >= (dom['size'] / 2.0):
            valid_indices.append(i)
            total_valid_pigments += dom['size']

    if total_valid_pigments > 0:
        _tw = sum(domain_info[i]['mu2'] for i in valid_indices)
        print('[QY-WEIGHTED] x0 ~ sum |mu|^2; total mu2 = %.1f D^2' % _tw)
        for i in valid_indices: x0[i] = domain_info[i]['mu2'] / _tw
    else:
        pigment_counts = np.array([d['size'] for d in domain_info])
        x0[:num_domains] = pigment_counts / np.sum(pigment_counts)

    max_steps = int(SIMULATION_DURATION_PS / MARKOV_TIME_STEP_PS)
    xt = np.zeros((max_steps + 1, num_domains + 2)); xt[0] = x0
    P_T = P.T
    for t in range(1, max_steps + 1):
        xt[t] = xt[t-1] @ P_T
        if np.sum(xt[t, :num_domains]) < SIMULATION_STOP_THRESHOLD: return xt[:t+1]
    return xt

# ==============================================================================
#  6. MAIN
# ==============================================================================

if __name__ == "__main__":
    # --- v5 FIX 2: provenance banner -------------------------------------------------
    # v4 run dirs symlinked ../eng.py and echoed almost none of their settings, so a
    # directory could not be re-run to its own logged number.  engine md5 + the list of
    # explicitly-set knobs is a complete determinant of the result (defaults are frozen
    # by the md5).  Also written to provenance.json beside run.log.
    _KNOBS = ["TIME_STEP_FS", "MU2_A", "V_CUTOFF", "R_C_ANG", "K_CS",
              "VIBRONIC", "VIB_WMIN", "S_RC", "S_ANT", "S_PER_GROUP", "V4_LINESHAPE_COMPAT",
              "PV_WMAX_CM", "PV_EPSREL", "PV_LIMIT", "PV_GAMMA_CUT", "PV_FULL_RANGE",
              "USE_PARTICIPATION", "CHLB_E", "CHLC_E", "CHLC_DIP2",
              "USE_TRESP", "CHLC_AC_SCALE", "TRESP_SCALE", "CHLC_TARGET_D",
              "PR_DOMAINS", "PR_FWHM", "PR_NREAL", "PR_CUT", "PR_SEED", "PDB_FILE", "RC_IDS",
              "DISORDER_FWHM", "DISORDER_SEED",
              "BELT_FUNNEL", "FUNNEL_STEP", "P700_E",
              "REDFIELD_RENORM_E", "LOC_VIB", "LINESHAPE_GAMMA",
              "AC_CALIB", "AC_TAU_FS", "RC_SINK_MODE", "RC_SINK_RANK",
              "DB_ENFORCE", "V4_DUMP", "V4_DUMP_FILE", "LEGACY_DOMAIN_ME"]
    import hashlib as _hl, json as _js, sys as _sys
    _self = os.path.realpath(__file__)
    try:
        _md5 = _hl.md5(open(_self, "rb").read()).hexdigest()
    except Exception:
        _md5 = "unavailable"
    _set = {_k: os.environ[_k] for _k in _KNOBS if _k in os.environ}
    _dflt = sorted(_k for _k in _KNOBS if _k not in os.environ)
    print("[ENGINE] %s" % _self)
    print("[ENGINE] md5=%s  python=%s  numpy=%s" % (_md5, _sys.version.split()[0], np.__version__))
    print("[SETTINGS] set: %s" % (" ".join("%s=%s" % _kv for _kv in sorted(_set.items()))
                                  or "(none - every knob at its default)"))
    print("[SETTINGS] at default: %s" % " ".join(_dflt))
    try:
        _js.dump({"engine": _self, "engine_md5": _md5,
                  "python": _sys.version.split()[0], "numpy": np.__version__,
                  "set": _set, "at_default": _dflt},
                 open("provenance.json", "w"), indent=1)
    except Exception as _e:
        print("[SETTINGS] could not write provenance.json: %s" % _e)

    print("[STRUCTURE] %s   RC pigments: %s" % (PDB_FILENAME, RC_IDS))
    if DISORDER_FWHM_CM > 0:
        if _DIS_SEED >= 0:
            np.random.seed(_DIS_SEED)
        print("[DISORDER] site-energy sigma = %.1f cm-1 (FWHM %.1f), seed=%s, %d realization(s)"
              % (DISORDER_FWHM_CM / 2.35482, DISORDER_FWHM_CM,
                 _DIS_SEED if _DIS_SEED >= 0 else "unseeded", N_MONTE_CARLO_RUNS))
        print("[DISORDER] average tau ACROSS independent runs; one run = one realization")
    else:
        print("[DISORDER] none (single realization, sigma = 0)")
    print("--- Step 1: Deterministic Setup ---")
    chlorophylls = read_structure(PDB_FILENAME)
    site_data = read_site_energies(SITE_ENERGY_FILENAME)
    chlorophylls = assign_site_energies(chlorophylls, site_data)
    # --- ANT_GAP: core/antenna site-energy offset.  DEFAULT +65 cm-1 on antenna Chl a --
    # ⛔ 65, NOT 150.  Verified against the primary sources 2026-08-23:
    #  * Akhtar 2025 Cell Rep Phys Sci 6, 102731, MAIN TEXT p.3 (their model input):
    #    "The Chl site energies were taken at random within normal distributions, centered
    #    around 675 and 678 nm for FCPI and PSI, respectively."  675 nm = 14814.8, 678 nm =
    #    14749.3  ->  antenna +65.6 cm-1 ABOVE core.  This is the value used here.
    #  * Their SUPPLEMENTAL METHODS states the SAME number pair with the labels SWAPPED
    #    ("14800 cm-1 for the PSI core and 14750 cm-1 for FCPI" -> antenna 50 BELOW).  The
    #    main-text reading is the self-consistent one: their own MEASURED core maximum is
    #    679 nm, which matches "678 nm for PSI" to 1 nm, whereas 14800 = 675.7 nm would
    #    invert the ordering they measured.
    #  * 153 cm-1 (672 vs 679 nm) is ISOLATED FCP vs ISOLATED core, and the same page says
    #    the BOUND antenna "has ... a strongly red-shifted Qy absorption band that overlaps
    #    with the absorption of the core complex".  Do not use it for a bound antenna.
    #  * Betti & Cupellini 2026 JACS Au, plant PSI-LHCI QM/MMPol (their Figs S3/S4):
    #    Lhca Chl a  16692.8 +- 138.9 (n=43) vs core Chl a 16641.7 +- 143.3 (n=100)
    #    -> +51 cm-1 for Chl a alone.  Including their Chl b the whole antenna is +147 --
    #    which is where "150" comes from.  Chl c is carried explicitly here (CHLC_E), so
    #    only the Chl a term belongs in ANT_GAP; using 150 double-counts the accessory pool.
    # UNIFORM: the same offset is added to EVERY antenna Chl a.  There is no belt/layer
    # dependence -- see BELT_FUNNEL below, which is a separate, DISABLED (default 0) knob.
    # Core/antenna split is STRUCTURAL, not by CORE_CHAIN_IDS (which is lowercase-only and so
    # mislabels 1JB0 and 5ZGB): a chain is ANTENNA iff it holds no RC pigment AND has at least
    # ANT_MIN_CHL chlorophylls.  LHC subunits carry 8-19; small core subunits carry 1-4.
    # Cyanobacterial PSI has no antenna chain, so 1JB0 is untouched by construction.
    # An explicit value in site_energies.txt always wins - the gap is never added on top.
    _agap   = float(os.environ.get("ANT_GAP", "65"))
    _agap_c = float(os.environ.get("ANT_GAP_CHLC", "0"))
    _minant = int(os.environ.get("ANT_MIN_CHL", "7"))
    if _agap or _agap_c:
        _per = {}
        for _c in chlorophylls: _per.setdefault(_c["chain_id"], []).append(_c)
        _rcch = {k[0] for k in RC_IDS}
        _antch = sorted(ch for ch, v in _per.items() if ch not in _rcch and len(v) >= _minant)
        _na = _nc = _skip = 0
        for _c in chlorophylls:
            if _c["chain_id"] not in _antch: continue
            if site_data and (_c["chain_id"], _c["res_seq"]) in site_data: _skip += 1; continue
            _pt = _c.get("ptype", "a")
            if   _pt == "a" and _agap:   _c["site_energy"] += _agap;   _na += 1
            elif _pt == "c" and _agap_c: _c["site_energy"] += _agap_c; _nc += 1
        print("[ANT-GAP] %d antenna chains %s | Chl a %+.0f cm-1 on %d pigments | Chl c %+.0f on %d"
              % (len(_antch), _antch, _agap, _na, _agap_c, _nc))
        if _skip:
            print("[ANT-GAP] %d antenna pigments skipped - explicit site_energies.txt value wins" % _skip)
    else:
        print("[ANT-GAP] disabled (ANT_GAP=0)")
    chlorophylls = calculate_centers_and_dipoles(chlorophylls)
    chlorophylls = build_tresp_clouds(chlorophylls)
    chlorophylls = _setup_chlc_tresp(chlorophylls)
    # --- v5 FIX 3: honest label.  This is EVERY chain, core included - not "non-core".
    # x0 is then Qy-dipole-weighted over all domains, i.e. whole-complex excitation.
    INITIAL_EXCITATION_CHAINS = {c['chain_id'] for c in chlorophylls}
    _ncore_ch = len([_c for _c in INITIAL_EXCITATION_CHAINS if _c in CORE_CHAIN_IDS])
    print("[EXCITATION] whole-complex: all %d pigment chains (%d core + %d antenna); "
          "x0 ~ |mu|^2" % (len(INITIAL_EXCITATION_CHAINS), _ncore_ch,
                           len(INITIAL_EXCITATION_CHAINS) - _ncore_ch))

    # --- NEW: Identify Red Forms ---
    # --- RED FORMS: assigned ONLY from site_energies.txt; geometric rule removed ---
    _redkeys = set(site_data.keys()) if site_data else set()
    _nr = 0
    for _c in chlorophylls:
        _c["is_red"] = (_c["chain_id"], _c["res_seq"]) in _redkeys
        if _c["is_red"]:
            _nr += 1
    print("--- REDS from site_energies.txt: %d pigments (geometric rule removed) ---" % _nr)

    import os as _osf, numpy as _npf
    if _osf.environ.get("BELT_FUNNEL","0")=="1":
        _step=float(_osf.environ.get("FUNNEL_STEP","50"))
        _core=_npf.array([c["mg_coords"] for c in chlorophylls if c["chain_id"] in CORE_CHAIN_IDS and c.get("mg_coords") is not None])
        _ch={}
        for _c in chlorophylls:
            if _c["chain_id"] not in CORE_CHAIN_IDS and _c.get("mg_coords") is not None:
                _ch.setdefault(_c["chain_id"],[]).append(_c["mg_coords"])
        _ch={_k:_npf.array(_v) for _k,_v in _ch.items()}
        _belt={}; _assigned=_core; _remaining=set(_ch); _b=1
        while _remaining:
            _cur=[_x for _x in _remaining if _npf.min(_npf.linalg.norm(_ch[_x][:,None,:]-_assigned[None,:,:],axis=2))<30.0]
            if not _cur: _cur=list(_remaining)
            for _x in _cur: _belt[_x]=_b; _remaining.discard(_x)
            _assigned=_npf.vstack([_assigned]+[_ch[_x] for _x in _cur]); _b+=1
        _nf=0
        for _c in chlorophylls:
            if _c["chain_id"] in CORE_CHAIN_IDS or _c.get("is_red",False): continue
            if _c["chain_id"] in _belt:
                _c["site_energy"]=_c["site_energy"]+_step*_belt[_c["chain_id"]]; _nf+=1
        print("--- BELT-FUNNEL(floodfill 30A): +%g/belt to %d non-core non-red Chls; max belt=%d ---"%(_step,_nf,(max(_belt.values()) if _belt else 0)))
    import os as _o
    # ⛔ P700_E IS AN OVERRIDE SWITCH, NOT AN ENERGY.  P700_E=0 does NOT put P700 at 0 cm-1 --
    # it DISABLES the override so P700 keeps whatever site_energies.txt (or the default) gave it.
    # Any run that supplies P700 energies via site_energies.txt MUST pass P700_E=0, otherwise
    # they are silently overwritten here, AFTER the file has been read.  Both branches now log.
    _p7 = float(_o.environ.get("P700_E", "14800"))
    _rcnow = [(_c["chain_id"], _c["res_seq"], _c["site_energy"]) for _c in chlorophylls
              if (_c["chain_id"], _c["res_seq"]) in RC_IDS]
    if _p7 > 0:
        print("[P700] OVERRIDE ON (P700_E=%.1f): %s -> all forced to %.1f cm-1"
              % (_p7, ", ".join("%s:%d was %.1f" % r for r in _rcnow), _p7))
        for _c in chlorophylls:
            if (_c["chain_id"], _c["res_seq"]) in RC_IDS: _c["site_energy"] = _p7
        print("--- P700 set to %.0f ---" % _p7)
    else:
        print("[P700] OVERRIDE DISABLED (P700_E=0) - P700 keeps its own site energies: %s"
              % ", ".join("%s:%d = %.1f cm-1" % r for r in _rcnow))
    v_mn_global, h_ex_mean = calculate_hamiltonian(chlorophylls, stochastic=False)
    if os.environ.get("PR_DOMAINS", "1") == "1":
        exciton_domains_indices = _pr_domains(v_mn_global, h_ex_mean, chlorophylls)
    else:
        exciton_domains_indices = partition_into_exciton_domains(v_mn_global, h_ex_mean, V_CUTOFF, SITE_ENERGY_DIFF_CUTOFF)

    # --- RC-MERGE: both P700 Chls MUST end in one domain -----------------------------
    # The PR clustering can leave the special pair split when its Coulomb coupling is
    # weaker than each P700's coupling to its own accessory neighbours (C. merolae:
    # 5.643 A Mg-Mg but only ~8.6 cm-1 TrEsp).  A split pair silently DOUBLES the total
    # charge-separation capacity, because each half-domain gets its own K_CS sink.
    # Merging is the fix; a PR_CUT scan is not (it only merges once the domain has
    # swallowed a third of the complex).  Default ON: it is a no-op for every structure
    # whose pair already clusters, so no settled number moves.
    if os.environ.get("RC_MERGE", "1") == "1":
        _rcdoms = [_i for _i, _d in enumerate(exciton_domains_indices)
                   if any((chlorophylls[_g]["chain_id"], chlorophylls[_g]["res_seq"]) in RC_IDS
                          for _g in _d)]
        if len(_rcdoms) > 1:
            _oldsz = [len(exciton_domains_indices[_i]) for _i in _rcdoms]
            _merged = sorted(_g for _i in _rcdoms for _g in exciton_domains_indices[_i])
            _keep = [_d for _i, _d in enumerate(exciton_domains_indices) if _i not in _rcdoms]
            exciton_domains_indices = [_merged] + _keep
            print("[RC-MERGE] P700 was split across %d domains (sizes %s) -> merged into one "
                  "domain of %d pigments" % (len(_rcdoms), _oldsz, len(_merged)))
    # --- RC-CHECK: hard gate.  Every run must report exactly one RC domain, 2 RC Chls.
    _rcdoms = [_i for _i, _d in enumerate(exciton_domains_indices)
               if any((chlorophylls[_g]["chain_id"], chlorophylls[_g]["res_seq"]) in RC_IDS
                      for _g in _d)]
    _nrc_in = sum(1 for _i in _rcdoms for _g in exciton_domains_indices[_i]
                  if (chlorophylls[_g]["chain_id"], chlorophylls[_g]["res_seq"]) in RC_IDS)
    _ok = (len(_rcdoms) == 1 and _nrc_in == len(RC_IDS))
    print("[RC-CHECK] %s  n_RC_domains=%d  RC_pigments_found=%d/%d  RC_domain_size=%s"
          % ("PASS" if _ok else "*** FAIL ***", len(_rcdoms), _nrc_in, len(RC_IDS),
             [len(exciton_domains_indices[_i]) for _i in _rcdoms]))
    if not _ok:
        print("[RC-CHECK] *** the total charge-separation capacity is NOT K_CS - this run is "
              "NOT comparable to the others.  Set RC_MERGE=1 or fix RC_IDS. ***")
        if os.environ.get("RC_CHECK_STRICT", "0") == "1":
            raise SystemExit("[RC-CHECK] aborting: RC pigments are not in a single domain")
    _sz = sorted((len(_d) for _d in exciton_domains_indices), reverse=True)
    _rcsz = []
    for _d in exciton_domains_indices:
        if any((chlorophylls[_g]["chain_id"], chlorophylls[_g]["res_seq"]) in RC_IDS for _g in _d):
            _rcsz.append(len(_d))
    # --- v5 FIX 4: name the partition actually used.  v4 printed V_CUTOFF unconditionally,
    # including when PR clustering produced the domains and V_CUTOFF played no part; and it
    # never printed PR_SEED, which changes n_domains 102 <-> 99 and tau by 0.5%.
    if os.environ.get("PR_DOMAINS", "1") == "1":
        _psrc = ("PR-clustering(FWHM=%s nreal=%s cut=%s seed=%s)"
                 % (os.environ.get("PR_FWHM", "180"), os.environ.get("PR_NREAL", "400"),
                    os.environ.get("PR_CUT", "0.8"), os.environ.get("PR_SEED", "70000")))
    else:
        _psrc = "V_CUTOFF=%.1f,dE<=%.0f" % (V_CUTOFF, SITE_ENERGY_DIFF_CUTOFF)
    print("[DOMAINS] partition=%s  n_domains=%d  largest=%d  top5=%s  RC_domain_sizes=%s  n_monomers=%d"
          % (_psrc, len(exciton_domains_indices), _sz[0], _sz[:5], sorted(_rcsz),
             sum(1 for _x in _sz if _x == 1)))
    with open(DOMAIN_FILE, "w") as f:
        f.write("# Domain Info\n")
        for i, idxs in enumerate(exciton_domains_indices): f.write(f"Domain {i}: {len(idxs)} pigments\n")

    domain_info, rc_indices = load_domain_info(exciton_domains_indices, chlorophylls)
    sum_domain_rates = np.zeros((len(domain_info), len(domain_info)))
    sum_trapping_rates = np.zeros(len(domain_info))

    print("\nPre-calculating G(t) and Lambda...")
    t_lineshape_axis = np.arange(0, TIME_MAX_FS, TIME_STEP_FS)
    # Added G(t) and Lambda for Red forms
    g_t_core = precalculate_g_function(t_lineshape_axis, HUANG_RHYS_S_CORE, TEMP_K)
    g_t_ant  = precalculate_g_function(t_lineshape_axis, HUANG_RHYS_S_ANT, TEMP_K)
    g_t_red  = precalculate_g_function(t_lineshape_axis, HUANG_RHYS_S_RED, TEMP_K)

    _G_VIB = precalculate_g_function(t_lineshape_axis, 0.0, TEMP_K)
    _G_PH1 = precalculate_g_function(t_lineshape_axis, 1.0, TEMP_K) - _G_VIB
    print('[P700-S] phonon S: %.2f on the P700 Chls (lambda=%.1f cm-1), %.2f elsewhere (%.1f cm-1)'
          % (HUANG_RHYS_S_RC, precalculate_lambda(HUANG_RHYS_S_RC),
             HUANG_RHYS_S_ANT, precalculate_lambda(HUANG_RHYS_S_ANT)))
    lambda_core = precalculate_lambda(HUANG_RHYS_S_CORE)
    lambda_ant  = precalculate_lambda(HUANG_RHYS_S_ANT)
    lambda_red  = precalculate_lambda(HUANG_RHYS_S_RED)

    print(f"\n--- Step 2: Running {N_MONTE_CARLO_RUNS} Monte Carlo Steps ---")
    omega_range_cm = np.linspace(13000, 16000, 501)

    for run in range(N_MONTE_CARLO_RUNS):
        print(f" Run {run+1}...")
        _, h_ex_stochastic = calculate_hamiltonian(chlorophylls, stochastic=True)
        domain_results = get_stochastic_domain_results(h_ex_stochastic, exciton_domains_indices, chlorophylls)

        # 1. Renormalize Energies
        renormalized_energies = []
        for i, d in enumerate(domain_results):
            renorm_E = calculate_renormalized_energies_full(d, chlorophylls, lambda_core, lambda_ant, lambda_red)
            renormalized_energies.append(renorm_E)

        # 2. Intradomain Rates
        _rEn = os.environ.get('REDFIELD_RENORM_E', '0') == '1'      # SETTLED: bare w, as Betti/pyQME
        all_redfield_rates = [calculate_redfield_rates(d, chlorophylls,
                              energies=(renormalized_energies[_q] if _rEn else None))
                              for _q, d in enumerate(domain_results)]
        print('[REDFIELD-E] %s exciton energies' % ('effective (v2)' if _rEn else 'bare'))

        all_abs_shapes = []
        all_ems_shapes = []
        _TD = []

        # 3. Lineshapes
        for i, d in enumerate(domain_results):
            n_exc = len(d['local_energies'])
            d_abs = []; d_ems = []
            _P = {'E': [], 'g': [], 'deph': []}

            # DEAD (v5 note): g_t_core / g_t_red / g_t_ant were selected here and then
            # unconditionally overwritten below in v4.  Kept only so the flags stay readable.
            is_red = any(chlorophylls[gx]['is_red'] for gx in d['global_indices'])
            is_p700 = any(chlorophylls[gx]['is_rc'] for gx in d['global_indices'])

            # --- v5 FIX 1: S weighting of the exciton lineshape ------------------------
            # pyQME builds g_a = Sum_Z w_aaaa[Z] g_Z with w_aaaa[Z] = Sum_{i in Z}|c_ia|^4,
            # so the phonon weight is Sum_i c_ia^4 S_i == _gamma_S(d)[a,a].
            # v4 used gamma_aa * S_a = (Sum_i c_ia^4)(Sum_i c_ia^2 S_i), weighting S twice.
            # The two are identical wherever S is uniform over the domain, so this moves
            # only the excitons that mix P700 (S=1) with antenna (S=0.5).
            # S_PER_GROUP=0 or V4_LINESHAPE_COMPAT=1 restores the v4 form bit-for-bit.
            _ls_compat = os.environ.get('V4_LINESHAPE_COMPAT', '0') == '1'
            _spg_ls    = os.environ.get('S_PER_GROUP', '1') == '1'
            _gS_ls     = _gamma_S(d, chlorophylls)
            _S_ls      = _S_per_exciton(d, chlorophylls)
            _gamma_mat = calculate_gamma(d, chlorophylls)   # alpha-independent: hoisted
            _PH = _G_PH1 - _G_PH1[0]
            _VB = _G_VIB - _G_VIB[0]

            for alpha in range(n_exc):
                k_out = np.sum(all_redfield_rates[i][alpha, :])

                if k_out > 1e-10:
                    lifetime_est_ps = 1.0 / k_out
                else:
                    lifetime_est_ps = 10000.0

                energy_cm = renormalized_energies[i][alpha]
                gamma_aa = _gamma_mat[alpha, alpha]
                _w_ph = (gamma_aa * _S_ls[alpha]) if (_ls_compat or not _spg_ls) \
                        else _gS_ls[alpha, alpha]

                _P['E'].append(energy_cm)
                if os.environ.get('LOC_VIB','1') == '1':                    # SETTLED: cR loc. vib.
                    # cR loc. vib. (Saraceno 2026): high-frequency modes stay LOCALIZED on the
                    # molecule, so the sideband must NOT be suppressed by the exciton IPR.
                    _P['g'].append(_w_ph * _PH + _VB)
                else:
                    # the modes carry their own molecular Huang-Rhys factors from the table,
                    # so they take the bare IPR weight gamma_aa and NOT the phonon S.
                    _P['g'].append(_w_ph * _PH + gamma_aa * _VB)
                # Re(gamma) = lifetime broadening, Gamma_a = 0.5*sum_b k_{a->b} (Kim/Akhtar Eq.7,
                # = Re of pyQME's redf_dephasing).  Off by default: it is a symmetric Lorentzian
                # and breaks detailed balance.  LINESHAPE_GAMMA=1 restores the Betti/Cupellini form.
                _P['deph'].append(0.5 * k_out / 1000.0
                                  if os.environ.get('LINESHAPE_GAMMA','1') == '1' else 0.0)   # SETTLED: keep Re(gamma)
            _TD.append(_P)
            all_abs_shapes.append(d_abs); all_ems_shapes.append(d_ems)

        # 4. Forster Rates
        all_forster_rates = {}; _ALLV = {}
        for i in range(len(domain_results)):
            for j in range(len(domain_results)):
                if i == j: continue
                dist = get_min_domain_distance(domain_results[i], domain_results[j], chlorophylls)
                if dist < INTER_DOMAIN_DISTANCE_CUTOFF:
                    V_eff = calculate_interdomain_coupling(domain_results[i], domain_results[j], v_mn_global)
                    rates = calculate_gf_rates_timedomain(V_eff, _TD[i], _TD[j], t_lineshape_axis)
                    all_forster_rates[(i, j)] = rates
                    _ALLV[(i, j)] = V_eff
        if os.environ.get("AC_CALIB", "0") == "1":
            # ---- Chl a <-> Chl c handled by a MEASURED escape rate, |V|^2-scaled, DB-consistent.
            # Rationale: the ONLY channel that needs the 69 modes is the 986 cm-1 a/c gap (131x).
            # Keep TrEsp |V|^2 (all distance/orientation info); replace the broken spectral-overlap
            # factor O by a single scalar calibrated so the mean Chl c escape time = AC_TAU_FS.
            _tauf = float(os.environ.get("AC_TAU_FS", "60.0"))
            _dtype = ["c" if any(chlorophylls[_g].get("ptype", "a") == "c"
                                 for _g in _d["global_indices"]) else "a"
                      for _d in domain_results]
            _acp = [(_i, _j) for (_i, _j) in all_forster_rates if _dtype[_i] != _dtype[_j]]
            # exciton dipole strengths (for weighting the calibration)
            _mu2 = {}
            for _i, _d in enumerate(domain_results):
                _C = np.asarray(_d["local_coefficients"])
                _mv = np.array([chlorophylls[_g]["dipole_vec"] for _g in _d["global_indices"]])
                _mu2[_i] = np.array([float(np.dot((_C[:, _a][:, None] * _mv).sum(axis=0),
                                                  (_C[:, _a][:, None] * _mv).sum(axis=0)))
                                     for _a in range(_C.shape[1])])
            # unit-overlap rates for every a<->c pair
            _unit = {}
            for (_i, _j) in _acp:
                _V = _ALLV[(_i, _j)]
                _unit[(_i, _j)] = 2.0 * ((_V / HBAR) ** 2) * 1000.0
            # calibrate on the c -> a direction
            _num = 0.0; _den = 0.0
            for _i, _d in enumerate(domain_results):
                if _dtype[_i] != "c": continue
                for _a in range(len(_d["local_energies"])):
                    _k1 = sum(_unit[(_i, _j)][_a, :].sum() for (_x, _j) in _acp if _x == _i)
                    if _k1 <= 0: continue
                    _w = _mu2[_i][_a]
                    _num += _w / _k1; _den += _w
            if _den > 0 and _num > 0:
                _scale = (_num / _den) / (_tauf / 1000.0)
                _kT = KB * TEMP_K
                _nrep = 0
                for (_i, _j) in _acp:
                    _R = np.zeros_like(all_forster_rates[(_i, _j)])
                    for _a in range(_R.shape[0]):
                        for _b in range(_R.shape[1]):
                            _kdown = _scale * _unit[(_i, _j)][_a, _b]
                            _dE = renormalized_energies[_j][_b] - renormalized_energies[_i][_a]
                            # downhill if acceptor is lower; otherwise DB-scale from the reverse
                            _R[_a, _b] = _kdown if _dE <= 0.0 else _kdown * np.exp(-_dE / _kT)
                    all_forster_rates[(_i, _j)] = _R
                    _nrep += 1
                _cesc = (_num / _den) / _scale
                print("[AC-CALIB] %d a<->c domain pairs rescaled; overlap scale=%.4e ; "
                      "mean Chl c escape %.1f fs (target %.1f)"
                      % (_nrep, _scale, 1000.0 * _cesc, _tauf))
            else:
                print("[AC-CALIB] no a<->c pairs found - nothing done")
        domain_rates_i = calculate_domain_level_rates(domain_results, all_forster_rates, rc_indices)
        sum_domain_rates += domain_rates_i

        # 5. Weighted Trapping Rates
        trapping_rates_i = calculate_weighted_trapping_rates(domain_results, chlorophylls, CHARGE_SEP_RATE_PS, TEMP_K)
        sum_trapping_rates += trapping_rates_i

    avg_rates = sum_domain_rates / N_MONTE_CARLO_RUNS
    avg_trapping_rates = sum_trapping_rates / N_MONTE_CARLO_RUNS

    # --- v5 FIX 6: single db_dump.npz write.  v4 wrote this file TWICE; the second write
    # silently dropped emin / n_core / n_red, so anything reading those got nothing.
    # Merged here: the richer field set PLUS n_rc from the second block.
    import numpy as _np
    _n = len(domain_info)
    _size = _np.array([d['size'] for d in domain_info], float)
    _emin = _np.array([min(renormalized_energies[i]) for i in range(_n)], float)
    _eall = [list(map(float, renormalized_energies[i])) for i in range(_n)]
    _na = _np.zeros(_n); _nc = _np.zeros(_n); _core = _np.zeros(_n)
    _red = _np.zeros(_n); _rc = _np.zeros(_n)
    for i, d in enumerate(domain_results):
        for gx in d['global_indices']:
            c = chlorophylls[gx]
            if c.get('ptype', 'a') == 'c': _nc[i] += 1
            else: _na[i] += 1
            if c['chain_id'] in CORE_CHAIN_IDS: _core[i] += 1
            if c.get('is_red', False): _red[i] += 1
            if c.get('is_rc', False): _rc[i] += 1
    _mx = max(len(e) for e in _eall)
    _np.savez("db_dump.npz", rates=avg_rates, trap=avg_trapping_rates, size=_size, emin=_emin,
              n_a=_na, n_c=_nc, n_core=_core, n_red=_red, n_rc=_rc,
              eall=_np.array([_np.pad(_np.array(e), (0, _mx-len(e)), constant_values=_np.nan) for e in _eall]))
    print(f"[DBDUMP] n_domains={_n} Chl a={_na.sum():.0f} Chl c={_nc.sum():.0f} "
          f"(single write; fields: rates trap size emin n_a n_c n_core n_red n_rc eall)")

    _SINK_MODE = os.environ.get("RC_SINK_MODE", "lowest")
    print("[RC-SINK] mode=%s  K_CS=%.4f ps^-1" % (_SINK_MODE, CHARGE_SEP_RATE_PS))
    _idx = {}; _ne = 0
    for _i, _d in enumerate(domain_results):
        for _a in range(len(_d["local_energies"])):
            _idx[(_i, _a)] = _ne; _ne += 1
    _CS, _DI = _ne, _ne + 1
    _K = np.zeros((_ne + 2, _ne + 2))
    for _i, _d in enumerate(domain_results):
        _R = all_redfield_rates[_i]
        for _a in range(_R.shape[0]):
            for _b in range(_R.shape[1]):
                if _a != _b:
                    _K[_idx[(_i, _b)], _idx[(_i, _a)]] += _R[_a, _b]
    for _key in all_forster_rates:
        _i, _j = _key; _F = all_forster_rates[_key]
        for _a in range(_F.shape[0]):
            for _b in range(_F.shape[1]):
                _K[_idx[(_j, _b)], _idx[(_i, _a)]] += _F[_a, _b]
    if os.environ.get("DB_ENFORCE", "1") == "1":                # SETTLED: Betti enforcement
        _Eex0 = np.array([renormalized_energies[_i][_a]
                          for _i, _d in enumerate(domain_results)
                          for _a in range(len(_d["local_energies"]))])
        _beta = 1.0 / (KB * TEMP_K)
        # --- v5 FIX 5: count the pairs actually MODIFIED.  v4 printed the loop count
        # (every pair, ~n^2/2), which read as if enforcement had touched everything.
        _nfix = 0; _npair = 0
        for _x in range(_ne):
            for _y in range(_x + 1, _ne):
                _dEyx = _Eex0[_y] - _Eex0[_x]
                _f = np.exp(-_dEyx * _beta)
                if _dEyx > 0.0:
                    _old = _K[_y, _x]
                    _new = _K[_x, _y] * _f            # keep downhill y->x, rebuild uphill
                    _K[_y, _x] = _new
                else:
                    _old = _K[_x, _y]
                    _new = _K[_y, _x] / _f
                    _K[_x, _y] = _new
                if abs(_new - _old) > 1e-9 * max(abs(_new), abs(_old), 1e-300): _nfix += 1
                _npair += 1
        print("[DB-ENFORCE] downhill-anchored, ladder = E-tilde (renormalized): "
              "%d of %d exciton pairs modified" % (_nfix, _npair))
    _kR = np.zeros(_ne); _kF = np.zeros(_ne)
    for _i, _d in enumerate(domain_results):
        _R = all_redfield_rates[_i]
        for _a in range(_R.shape[0]):
            _kR[_idx[(_i, _a)]] = _R[_a, :].sum()
    for _key in all_forster_rates:
        _i, _j = _key; _F = all_forster_rates[_key]
        for _a in range(_F.shape[0]):
            _kF[_idx[(_i, _a)]] += _F[_a, :].sum()
    _mul = _kR > 0
    _rat = _kR[_mul] / np.maximum(_kF[_mul], 1e-30)
    print("[EXC-ME] %d excitons (%d in multi-exciton domains). k_Redfield/k_Forster: median %.1f p10 %.1f"
          % (_ne, _mul.sum(), np.median(_rat) if _mul.any() else 0.0,
             np.percentile(_rat, 10) if _mul.any() else 0.0))
    _Eex = np.array([renormalized_energies[_i][_a] for _i, _d in enumerate(domain_results)
                     for _a in range(len(_d["local_energies"]))])
    _intra = np.zeros((_ne, _ne), bool); _o = 0
    for _i, _d in enumerate(domain_results):
        _n = len(_d["local_energies"]); _intra[_o:_o+_n, _o:_o+_n] = True; _o += _n
    _KK = _K[:_ne, :_ne]
    _off = ~np.eye(_ne, dtype=bool)
    _mm = (_KK > 0) & (_KK.T > 0) & _off
    _onezero = _off & (((_KK > 0) & (_KK.T <= 0)) | ((_KK <= 0) & (_KK.T > 0)))
    print("[GF-CLAMP] %d rates evaluated; %d clamped to 0 (most negative %.3e ps^-1); %d clamped at 5000"
          % (_CLAMP['n'], _CLAMP['neg'], _CLAMP['negmin'], _CLAMP['hi']))
    print("[DB-MASK] %d ordered pairs excluded from DB-CHECK because one direction is exactly 0 "
          "(they carry %.3e ps^-1 of downhill rate, %.4f %% of all inter-exciton rate)"
          % (int(_onezero.sum()), float(_KK[_onezero].sum()),
             100.0*float(_KK[_onezero].sum())/max(float(_KK[_off][_KK[_off] > 0].sum()), 1e-30)))
    _ii, _jj = np.nonzero(_mm)
    _dE = _Eex[_jj] - _Eex[_ii]
    _kp = _dE > 0; _ii, _jj, _dE = _ii[_kp], _jj[_kp], _dE[_kp]
    _err = (_KK[_ii, _jj] / _KK[_jj, _ii]) / np.exp(_dE / (KB * TEMP_K)) - 1.0
    _isin = _intra[_ii, _jj]; _w = _KK[_ii, _jj]
    print("[DB-CHECK] k(D->A)/k(A->D) vs exp(dE/kT), %d downhill pairs:" % len(_err))
    for _lbl, _sel in (("Redfield (intra)", _isin), ("gen-Forster (inter)", ~_isin)):
        if _sel.sum() == 0: continue
        _e = np.abs(_err[_sel]); _ww = _w[_sel]
        print("   %-20s n=%-6d median %+9.5f %%   flux-weighted mean|err| %8.5f %%"
              % (_lbl, _sel.sum(), 100*np.median(_err[_sel]), 100*np.sum(_e*_ww)/_ww.sum()))
    if os.environ.get("V4_DUMP", "1") == "1":
        _bare = np.array([domain_results[_i]["local_energies"][_a]
                          for _i, _d in enumerate(domain_results)
                          for _a in range(len(_d["local_energies"]))])
        np.savez(os.environ.get("V4_DUMP_FILE", "v4_dump.npz"),
                 E_bare=_bare, E_tilde=_Eex, K=_KK, intra=_intra,
                 dom_sizes=np.array([len(_d["local_energies"]) for _d in domain_results]),
                 reorg=np.concatenate(_REORG_PARTS) if _REORG_PARTS else np.zeros(0),
                 disp=np.concatenate(_DISP_PARTS) if _DISP_PARTS else np.zeros(0),
                 pv_epsrel=float(os.environ.get('PV_EPSREL','1e-2')),
                 pv_cut=float(os.environ.get('PV_GAMMA_CUT','1e-4')),
                 dt_fs=TIME_STEP_FS, tmax_fs=TIME_MAX_FS,
                 clamp_neg=_CLAMP["neg"], clamp_hi=_CLAMP["hi"], clamp_n=_CLAMP["n"])
        print("[V4-DUMP] wrote %s" % os.environ.get("V4_DUMP_FILE", "v4_dump.npz"))
    for _i, _d in enumerate(domain_results):
        _hasrc = any((chlorophylls[_g]["chain_id"], chlorophylls[_g]["res_seq"]) in RC_IDS
                     for _g in _d["global_indices"])
        _rcloc = [_k for _k, _g in enumerate(_d["global_indices"])
                  if (chlorophylls[_g]["chain_id"], chlorophylls[_g]["res_seq"]) in RC_IDS]
        for _a in range(len(_d["local_energies"])):
            _K[_DI, _idx[(_i, _a)]] = INTRINSIC_DISSIPATION_RATE_PS
            if _hasrc:
                if _SINK_MODE == "participation":
                    _pw = float(np.sum(np.asarray(_d["local_coefficients"])[_rcloc, _a] ** 2))
                    _K[_CS, _idx[(_i, _a)]] = CHARGE_SEP_RATE_PS * _pw
                elif _SINK_MODE == "lowest":
                    # which energies rank the RC excitons: 'bare' matches Betti/pyQME
                    _rank = (_d['local_energies'] if os.environ.get('RC_SINK_RANK','bare') == 'bare'
                             else renormalized_energies[_i])
                    if _a == int(np.argmin(_rank)):
                        _K[_CS, _idx[(_i, _a)]] = CHARGE_SEP_RATE_PS
                else:
                    _K[_CS, _idx[(_i, _a)]] = CHARGE_SEP_RATE_PS
    print("[RC-SINK] total charge-separation capacity = %.4f ps^-1 (should be %.4f = K_CS x n_P700)"
          % (_K[_CS, :_ne].sum(), CHARGE_SEP_RATE_PS * len(RC_IDS)))
    # --- v5 FIX 7: the 500 ps^-1 ceiling was applied silently, AFTER DB enforcement, so it
    # could have broken detailed balance without a word.  Measured over all 42 v4 dumps the
    # largest off-diagonal element is 28.24 ps^-1, i.e. it never fires - but now it says so.
    _nclip500 = int((_K > 500.0).sum())
    if _nclip500:
        print("[K-CLIP] %d elements exceeded the 500 ps^-1 ceiling and were clipped "
              "(max %.3f) - THIS BREAKS THE ENFORCED DETAILED BALANCE"
              % (_nclip500, float(_K.max())))
    else:
        print("[K-CLIP] none (largest element %.3f ps^-1, ceiling 500)" % float(_K.max()))
    _K = np.clip(_K, 0.0, 500.0)
    np.fill_diagonal(_K, 0.0)
    for _c in range(_ne + 2): _K[_c, _c] = -np.sum(_K[:, _c])
    _x0 = np.zeros(_ne + 2); _tot = 0.0
    for _i, _d in enumerate(domain_results):
        _C = np.asarray(_d["local_coefficients"])
        _mu = np.array([chlorophylls[_g]["dipole_vec"] for _g in _d["global_indices"]])
        _sz = len(_d["global_indices"])
        _mc = sum(1 for _g in _d["global_indices"]
                  if chlorophylls[_g]["chain_id"] in INITIAL_EXCITATION_CHAINS)
        if _mc < _sz / 2.0: continue
        for _a in range(_C.shape[1]):
            _v = (_C[:, _a][:, None] * _mu).sum(axis=0)
            _m2 = float(np.dot(_v, _v))
            _x0[_idx[(_i, _a)]] = _m2; _tot += _m2
    _x0 = _x0 / _tot
    _P = expm(_K * MARKOV_TIME_STEP_PS).T
    _steps = int(SIMULATION_DURATION_PS / MARKOV_TIME_STEP_PS)
    _h = [_x0]
    for _t in range(_steps):
        _h.append(_h[-1] @ _P)
        if np.sum(_h[-1][:_ne]) < SIMULATION_STOP_THRESHOLD: break
    _h = np.array(_h)
    _tp = np.arange(len(_h)) * MARKOV_TIME_STEP_PS
    _tau_exc = simpson(y=np.sum(_h[:, :_ne], axis=1), x=_tp)
    print("[EXC-ME] tau = %.6f ps   yield = %.6f   dissipation = %.6f"
          % (_tau_exc, _h[-1][_CS], _h[-1][_DI]))
    with open("exciton_me.out", "w") as _f:
        _f.write("Exciton_ME_tau_ps %.8f" % _tau_exc + chr(10)
                 + "Exciton_ME_yield %.8f" % _h[-1][_CS] + chr(10)
                 + "N_excitons %d" % _ne + chr(10))


    # --- v5 FIX 8: label the legacy v1 path.  Everything from here down is the v1
    # DOMAIN-level master equation.  Its outputs sit in the same run directory and are
    # easy to mistake for the result.  They are a different quantity.
    print("\n--- Step 3: LEGACY v1 DOMAIN-level master equation ---")
    print("[LEGACY-V1] What follows (Overall_Lifetime_ps, overall_system_analysis_MC_AVG.out,")
    print("[LEGACY-V1] excitation_decay.out, population_dynamics.*, ncf_summary.txt) is the")
    print("[LEGACY-V1] DOMAIN-level v1 result and is NOT the reported tau.")
    print("[LEGACY-V1] The tau of record is the exciton-level [EXC-ME] value above,")
    print("[LEGACY-V1] also written to exciton_me.out.  It also REWRITES domains.out in a")
    print("[LEGACY-V1] different format than the one written in Step 1.")
    history = run_simulation(avg_rates, avg_trapping_rates, domain_info)

    time_points = np.arange(len(history)) * MARKOV_TIME_STEP_PS
    exciton_pop = np.sum(history[:, :len(domain_info)], axis=1)
    np.savetxt("excitation_decay.out", np.column_stack((time_points, exciton_pop)), header="Time(ps) Population")

    CHARGE_SEP_IDX, DISSIPATION_IDX = len(domain_info), len(domain_info) + 1
    final_state = history[-1]
    yield_val = final_state[CHARGE_SEP_IDX]
    dissipation_val = final_state[DISSIPATION_IDX]
    lifetime_val = simpson(y=exciton_pop, x=time_points)

    print(f"\n--- Overall System Results ---")
    print(f"  Overall Yield:         {yield_val:.6f} ({yield_val*100.0:.2f} %)")
    print(f"  Overall Lifetime:      {lifetime_val:.6f} (ps)")
    print(f"  Overall Dissipation:   {dissipation_val:.6f} ({dissipation_val*100.0:.2f} %)")

    with open("overall_system_analysis_MC_AVG.out", "w") as f:
        f.write("Overall_Yield\t{:.8f}\n".format(yield_val))
        f.write("Overall_Lifetime_ps\t{:.8f}\n".format(lifetime_val))
        f.write("Overall_Dissipation\t{:.8f}\n".format(dissipation_val))
    print("Analysis summary saved to 'overall_system_analysis_MC_AVG.out'")
    
    # ==============================================================================
    #  PLOTTING AND POPULATION ANALYSIS
    # ==============================================================================
    print("\nGenerating population dynamics plot...")

    def get_domain_category(domain_chls):
        """
        Classifies domains based on the group with the highest pigment count.
        Priority in ties: EXCITED > REST > CORE.
        """
        # domain_chls is list of tuples: (chain_id, res_seq)
        count_excited = sum(1 for c_id, _ in domain_chls if c_id in INITIAL_EXCITATION_CHAINS)
        count_core = sum(1 for c_id, _ in domain_chls if c_id in CORE_CHAIN_IDS)
        total = len(domain_chls)
        count_rest = total - count_excited - count_core

        # Create a mapping
        counts = {
            "EXCITED": count_excited,
            "REST": count_rest,
            "CORE": count_core
        }

        # Find the maximum pigment count
        max_count = max(counts.values())

        # Identify which groups have this count (handling ties)
        winners = [cat for cat, val in counts.items() if val == max_count]

        # Tie-breaking priority: EXCITED > REST > CORE
        if "EXCITED" in winners:
            return "EXCITED"
        elif "REST" in winners:
            return "REST"
        return "CORE"

    # Calculate populations for each group
    pop_excited = np.zeros(len(time_points))
    pop_rest = np.zeros(len(time_points))
    pop_core = np.zeros(len(time_points))
    
    for i, dom in enumerate(domain_info):
        category = get_domain_category(dom['chlorophylls'])
        if category == "EXCITED":
            pop_excited += history[:, i]
        elif category == "CORE":
            pop_core += history[:, i]
        else:
            pop_rest += history[:, i]
            
    pop_yield = history[:, CHARGE_SEP_IDX]

    # --- Plotting ---
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    
    plt.figure(figsize=(8, 6))
    plt.plot(time_points, pop_excited, label='Initially excited chains', linewidth=2)
    plt.plot(time_points, pop_rest, label='Rest of Antenna', linewidth=2)
    plt.plot(time_points, pop_core, label='Core Complex', linewidth=2)
    plt.plot(time_points, pop_yield, label='Charge Separation (Yield)', linestyle='--', color='black')
    
    plt.xlabel('Time (ps)', fontsize=22)
    plt.ylabel('Population', fontsize=22)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Limit x-axis to 50ps (or max simulation time if shorter) for initial kinetics view
    plt.xlim(0, min(50, time_points[-1])) 
    
    plt.savefig('population_dynamics.png', dpi=300)
    plt.close()
    print("Saved 'population_dynamics.png'.")
    
    # Save Population Data to Text
    output_data = np.column_stack((time_points, pop_excited, pop_rest, pop_core, pop_yield))
    np.savetxt("population_dynamics.txt", output_data,
               header="Time(ps)\tExcited_Pop\tRestAntenna_Pop\tCore_Pop\tYield",
               fmt='%.6f', delimiter='\t')

    # ==============================================================================
    #  LEGACY EXPORTER: Generate files for scienceadvances2.py
    # ==============================================================================
    print("\n--- Exporting Legacy Formats for Visualization ---")

    # 1. Export 'domains.out' in the exact format parse_domains_file expects
    # Format:
    # ## Domain 1
    # # Chlorophylls
    # A 101
    # # Exciton Energies
    with open("domains.out", "w") as f:
        for i, dom in enumerate(domain_results):
            f.write(f"## Domain {i+1}\n")
            f.write("# Chlorophylls\n")
            for chl_tuple in dom['chlorophylls']:
                # chl_tuple is (chain_id, res_seq)
                f.write(f"{chl_tuple[0]} {chl_tuple[1]}\n")
            f.write("# Exciton Energies\n")
            f.write("0.00\n") # Placeholder to satisfy the parser

    # 2. Export 'ncf_summary.txt' with explicit forward/backward components
    print("\nCalculating Net Cumulative Flux with explicit forward/backward components...")

    num_doms = len(domain_results)
    pops = history[:, :num_doms]

    with open("ncf_summary.txt", "w") as f:
        f.write("Rank Sender Receiver ChainA ChainB GrossFwdFlux GrossBwdFlux NetFlux\n")
        rank = 1
        calculated_pairs = set()

        for i in range(num_doms):
            for j in range(num_doms):
                if i == j: continue
                pair_key = tuple(sorted((i, j)))
                if pair_key in calculated_pairs: continue

                k_i_to_j = avg_rates[j, i]
                k_j_to_i = avg_rates[i, j]

                flow_i_to_j = k_i_to_j * pops[:, i]
                flow_j_to_i = k_j_to_i * pops[:, j]

                gross_fwd_val = simpson(y=flow_i_to_j, dx=MARKOV_TIME_STEP_PS)
                gross_bwd_val = simpson(y=flow_j_to_i, dx=MARKOV_TIME_STEP_PS)

                net_val = gross_fwd_val - gross_bwd_val
                calculated_pairs.add(pair_key)

                if abs(net_val) > 0.001:
                    if net_val > 0:
                        sender, receiver = i, j
                        fwd, bwd, net = gross_fwd_val, gross_bwd_val, net_val
                    else:
                        sender, receiver = j, i
                        fwd, bwd, net = gross_bwd_val, gross_fwd_val, abs(net_val)

                    sender_str = f"D{sender+1}"
                    receiver_str = f"D{receiver+1}"
                    chain_a = domain_results[sender]['chlorophylls'][0][0]
                    chain_b = domain_results[receiver]['chlorophylls'][0][0]

                    f.write(f"{rank} {sender_str} -> {receiver_str} {chain_a} {chain_b} {fwd:.6e} {bwd:.6e} {net:.6e}\n")
                    rank += 1

# 3. Export 'ci_summary.txt' (Cumulative Input)
    # Formula: Integral[ Sum(k_ij * P_i(t)) for all i != j ] dt
    print("Calculating Cumulative Input (CI)...")
    
    ci_values = np.zeros(len(domain_results))
    
    # pops shape: (time_steps, num_domains)
    # avg_rates shape: (num_domains, num_domains) [Target, Source] -> rates[j, i] is i->j ?? 
    # WAIT: In your code: domain_rates[d2, d1] ... sum ... exciton_rates. 
    # Usually Matrix multiplication is P(t+1) = Q * P(t). 
    # Your code uses: xt[t] = xt[t-1] @ P_T (Row vector multiplication).
    # Therefore Q[row, col] usually implies rate from row->col or col->row depending on convention.
    # Looking at Q construction: Q[:num_domains, :num_domains] = avg_domain_rates
    # And calculate_domain_level_rates: domain_rates[d2, d1] = sum(...) -> Rate FROM d1 TO d2.
    # So avg_rates[j, i] is the rate FROM i TO j.
    
    for j in range(len(domain_results)): # j is the Receiver domain
        instantaneous_input = np.zeros(len(time_points))
        
        for i in range(len(domain_results)): # i is the Sender domain
            if i == j: continue
            
            rate_i_to_j = avg_rates[j, i]
            
            # Add flux from i -> j at every time point
            instantaneous_input += rate_i_to_j * pops[:, i]
            
        # Integrate total input over time
        ci_values[j] = simpson(y=instantaneous_input, dx=MARKOV_TIME_STEP_PS)

    with open("ci_summary.txt", "w") as f:
        f.write("Rank Domain Chain Value\n")
        for i, val in enumerate(ci_values):
            # Get chain of first pigment for label
            chain_label = domain_results[i]['chlorophylls'][0][0]
            dom_str = f"D{i+1}"
            f.write(f"{i+1} {dom_str} {chain_label} {val:.6e}\n")
    print("Successfully generated 'domains.out', 'ncf_summary.txt', and 'ci_summary.txt'.")
