import numpy as np
import os
import random
from math import sqrt, log, cos, pi, factorial
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
TIME_STEP_FS = 2.0
TIME_MAX_FS  = 5000.0

# Renger Parameters
RENGER_S1 = 0.8
RENGER_S2 = 0.5
RENGER_W1_CM = 0.56
RENGER_W2_CM = 1.94

# --- MODIFIED HUANG RHYS SECTION ---
HUANG_RHYS_S_CORE = 0.5
HUANG_RHYS_S_ANT  = 0.5 # Adjusted for bulk antenna
HUANG_RHYS_S_RED = 0.5
# -----------------------------------
# Controls
N_MONTE_CARLO_RUNS = 1
PDB_FILENAME = "1JB0.pdb"
SITE_ENERGY_FILENAME = "site_energies.txt"
DOMAIN_FILE = "domains.out"

INITIAL_EXCITATION_CHAINS = {'1','2','3','4','5'}  # all-antenna (default; pentamer-homolog needs alignment)

# Structure Params
# Chl a Qy dipole strength. Knox & Spring's 21.0 D^2 is the 0-0 value; it is rescaled
# to the full electronic strength further down if the vibronic modes are switched on.
MU2_CHLA_00 = 21.0
MAGNITUDE_MU_A = sqrt(MU2_CHLA_00)
REFRACTIVE_INDEX = 1.4
VMN_PREFACTOR = 5.04
C_FACTOR = (REFRACTIVE_INDEX**2 + 2.0)**2 / (9.0 * REFRACTIVE_INDEX**2)
DISORDER_FWHM_CM = 0# 80.0
SITE_ENERGY_DIFF_CUTOFF = 300.0
V_CUTOFF = 60.0
R_C = 5.0
INTER_DOMAIN_DISTANCE_CUTOFF = 100.0

# Energies
DEFAULT_PSI_CORE_ENERGY = 14800.0
DEFAULT_FCPI_ENERGY = 14800.0

CORE_CHAIN_IDS = {'A','B'}
RC_IDS = [('A', 1011), ('B', 1021)]
# Rates
import os as _oskc
CHARGE_SEP_RATE_PS = float(_oskc.environ.get('K_CS', '1.5'))
INTRINSIC_DISSIPATION_RATE_PS = 0.5 / 1000.0
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
# --- spectral-density switch -------------------------------------------------
SPECTRAL_DENSITY = _ov.environ.get("SPECTRAL_DENSITY", "vibronic").strip().lower()
if SPECTRAL_DENSITY not in ("vibronic", "phonon"):
    raise SystemExit("SPECTRAL_DENSITY must be 'vibronic' or 'phonon', got %r" % SPECTRAL_DENSITY)
VIB_MODES = []
if SPECTRAL_DENSITY == "vibronic":
    for _ln in open("chla_vibronic_TEA.txt"):
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#"):
            _w, _lam = float(_ln.split()[0]), float(_ln.split()[1])
            if _w >= 100.0: VIB_MODES.append((_w, _lam / _w))
    _SVIB = sum(_s for _, _s in VIB_MODES)
    print("[SD] vibronic: %d intramolecular modes >=100 cm-1, S_vib=%.3f, lambda_vib=%.1f cm-1"
          % (len(VIB_MODES), _SVIB, sum(_s*_w for _w, _s in VIB_MODES)))
else:
    _SVIB = 0.0
    print("[SD] phonon only: low-frequency B777 spectral density")
# the dipole convention follows the spectral density
MAGNITUDE_MU_A = sqrt(MU2_CHLA_00 / np.exp(-_SVIB))
print("[SD] Chl a dipole strength %.2f D^2 (0-0 value %.1f D^2, S_vib %.3f)"
      % (MAGNITUDE_MU_A**2, MU2_CHLA_00, _SVIB))

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

def _S_per_exciton(domain, chlorophylls):
    C = domain['local_coefficients']
    Sm = np.array([1.0 if chlorophylls[g].get('is_rc', False) else HUANG_RHYS_S_ANT
                   for g in domain['global_indices']])
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

def calculate_redfield_rates(domain, chlorophylls, temp_k=TEMP_K):
    num_excitons = len(domain['local_energies'])
    rates = np.zeros((num_excitons, num_excitons))
    gamma = calculate_gamma(domain, chlorophylls)

    # --- MODIFIED: CHECK FOR RED FORMS ---
    # We check if *any* pigment in this domain is flagged as red
    is_red_domain = any(chlorophylls[g_idx]['is_red'] for g_idx in domain['global_indices'])
    is_p700_domain = any(chlorophylls[g_idx]['is_rc'] for g_idx in domain['global_indices'])
    S_ex = _S_per_exciton(domain, chlorophylls)
    # -------------------------------------

    for alpha in range(num_excitons):
        for beta in range(num_excitons):
            if alpha == beta: continue
            omega_ab_cm = domain['local_energies'][alpha] - domain['local_energies'][beta]
            rates[alpha, beta] = calculate_redfield_rate_element(
                omega_ab_cm, gamma[alpha, beta], 0.5*(S_ex[alpha]+S_ex[beta]), temp_k
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
    _S_CUR = [S_ex[0] if len(S_ex) else HUANG_RHYS_S_ANT]
    # -------------------------------------

    renormalized_energies = np.zeros(num_excitons)
    W_MIN = 1.0 / HBAR
    W_MAX = 2000.0 / HBAR

    def integrand_numerator(w_radfs):
        if w_radfs <= 0: return 0.0
        j_w = spectral_density_raw(w_radfs, _S_CUR[0])
        n_w = bose_einstein(w_radfs, TEMP_K)
        return (w_radfs**2) * (1.0 + n_w) * j_w

    for alpha in range(num_excitons):
        _S_CUR[0] = S_ex[alpha]
        reorg_shift = -gamma[alpha, alpha] * (S_ex[alpha] * _lambda_unit())
        dispersive_shift = 0.0
        for beta in range(num_excitons):
            if alpha == beta: continue

            w_ab_radfs = (energies_cm[alpha] - energies_cm[beta]) / HBAR
            gamma_ab = gamma[alpha, beta]
            
            # OPTIMIZATION: Check if coupling is significant before integrating
            if abs(gamma_ab) < 1e-4: continue

            if W_MIN < w_ab_radfs < W_MAX:
                # OPTIMIZATION: Reduced limit from 50 to 20, added epsrel
                val, _ = quad(integrand_numerator, W_MIN, W_MAX,
                              weight='cauchy', wvar=w_ab_radfs, 
                              limit=20, epsrel=1e-2)
                val = -val
            else:
                def full_fraction(w):
                    denom = w_ab_radfs - w
                    if abs(denom) < 1e-12: return 0.0
                    return integrand_numerator(w) / denom
                
                # OPTIMIZATION: Reduced limit from 50 to 20, added epsrel
                val, _ = quad(full_fraction, W_MIN, W_MAX, 
                              limit=20, epsrel=1e-2)

            dispersive_shift += gamma_ab * val

        total_shift = reorg_shift + (dispersive_shift * HBAR)
        renormalized_energies[alpha] = energies_cm[alpha] + total_shift

    return renormalized_energies

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
                if (line.startswith("HETATM") and res_name_check in {"CLA", "CL0"}):
                    res_name, chain_id = line[17:20].strip(), line[21].strip()
                    res_seq = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    chl_key = (chain_id, res_seq)
                    if chl_key not in chls_dict:
                        is_rc = chl_key in RC_IDS
                        # Initialize 'is_red' as False, we will update it later
                        chls_dict[chl_key] = {"chain_id": chain_id, "res_seq": res_seq,
                                              "all_atoms": [], "atom_by_name": {}, "mg_coords": None,
                                              "nb_coords": None, "nd_coords": None,
                                              "is_rc": is_rc, "is_red": False}
                    chls_dict[chl_key]["all_atoms"].append(np.array([x, y, z]))
                    atom_name = line[12:16].strip()
                    chls_dict[chl_key]["atom_by_name"].setdefault(atom_name, np.array([x, y, z]))
                    if atom_name == 'MG': chls_dict[chl_key]["mg_coords"] = np.array([x, y, z])
                    elif atom_name == 'NB': chls_dict[chl_key]["nb_coords"] = np.array([x, y, z])
                    elif atom_name == 'ND': chls_dict[chl_key]["nd_coords"] = np.array([x, y, z])
    except FileNotFoundError:
        print(f"Error: PDB file '{filename}' not found."); exit()
    return list(chls_dict.values())

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
        else:
            chl["site_energy"] = DEFAULT_PSI_CORE_ENERGY if chl["chain_id"] in CORE_CHAIN_IDS else DEFAULT_FCPI_ENERGY
    return chlorophylls

def calculate_centers_and_dipoles(chlorophylls):
    for chl in chlorophylls:
        chl["center"] = chl["mg_coords"] if chl["mg_coords"] is not None else np.mean(chl["all_atoms"], axis=0)
        if chl["nb_coords"] is not None and chl["nd_coords"] is not None:
            vec = chl["nd_coords"] - chl["nb_coords"]
            norm = np.linalg.norm(vec)
            chl["dipole_vec"] = MAGNITUDE_MU_A * (vec / norm) if norm > 1e-6 else np.zeros(3)
        else: chl["dipole_vec"] = np.zeros(3)
    return chlorophylls

# ==============================================================================
#  TrEsp coupling option (Madjet, Abdurahman & Renger 2006, B3LYP Chl a charges).
#  Activated with env USE_TRESP=1. Uses the SAME scalar convention as the
#  point-dipole engine (116141 * C_FACTOR) so ONLY the multipole shape differs.
#  Charges are the unscaled B3LYP set (|mu|=5.63 D on 5ZGB geometry); TRESP_SCALE
#  rescales them to the manuscript dipole strength 21 D^2 (|mu|=sqrt(21)=4.583 D).
# ==============================================================================
USE_TRESP = os.environ.get("USE_TRESP", "0") == "1"
PREFACTOR_TRESP = 116141.0 * C_FACTOR          # cm^-1 * A / e^2 ; matches PDA far field
TRESP_SCALE = 0.8136 * (MAGNITUDE_MU_A / sqrt(MU2_CHLA_00))  # 0.8136 targets the 0-0 value
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

def build_tresp_clouds(chlorophylls):
    """Attach a neutralized, rescaled transition-charge cloud to each Chl."""
    n_full = 0
    for chl in chlorophylls:
        abn = chl.get("atom_by_name", {})
        names = [a for a in CHG_CHLA_RAW if a in abn]
        q = np.array([CHG_CHLA_RAW[a] * TRESP_SCALE for a in names])
        if q.size > 0:
            q = q - q.sum() / q.size          # enforce exact neutrality
        chl["tresp_q"] = q
        chl["tresp_xyz"] = np.array([abn[a] for a in names]) if names else np.zeros((0, 3))
        if q.size == len(CHG_CHLA_RAW):
            n_full += 1
    if USE_TRESP:
        print(f"[TrEsp] built charge clouds for {len(chlorophylls)} Chls "
              f"({n_full} complete); scale={TRESP_SCALE}, prefactor={PREFACTOR_TRESP:.1f}")
    return chlorophylls

def tresp_coupling(chl_m, chl_n):
    qm, Xm = chl_m["tresp_q"], chl_m["tresp_xyz"]
    qn, Xn = chl_n["tresp_q"], chl_n["tresp_xyz"]
    if qm.size == 0 or qn.size == 0:
        return 0.0
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
                if USE_TRESP:
                    v_temp = tresp_coupling(chlorophylls[m], chlorophylls[n])
                else:
                    r_vec_nm, R_nm = r_vec / 10.0, R / 10.0
                    mu_m, mu_n = chlorophylls[m]["dipole_vec"], chlorophylls[n]["dipole_vec"]
                    term1 = np.dot(mu_m, mu_n) / (R_nm**3)
                    term2 = 3.0 * np.dot(mu_m, r_vec_nm) * np.dot(mu_n, r_vec_nm) / (R_nm**5)
                    v_temp = VMN_PREFACTOR * C_FACTOR * (term1 - term2)
                if v_temp > 5000.0: v_temp = 5000.0
                if v_temp < -5000.0: v_temp = -5000.0
                v_mn[m, n] = v_mn[n, m] = h_ex[m, n] = h_ex[n, m] = v_temp
            else: v_mn[m, n] = v_mn[n, m] = 0.0
    return v_mn, h_ex

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
    print("--- Step 1: Deterministic Setup ---")
    chlorophylls = read_pdb_data(PDB_FILENAME)
    site_data = read_site_energies(SITE_ENERGY_FILENAME)
    chlorophylls = assign_site_energies(chlorophylls, site_data)
    chlorophylls = calculate_centers_and_dipoles(chlorophylls)
    chlorophylls = build_tresp_clouds(chlorophylls)
    # --- ALL-ANTENNA initial excitation: every non-core chain ---
    INITIAL_EXCITATION_CHAINS = {c['chain_id'] for c in chlorophylls}
    print(f"[WHOLE-PSI] exciting {len(INITIAL_EXCITATION_CHAINS)} chains (core + antenna)")


    # --- BELT-1-ONLY red rule (tightest dimer per inner-belt antenna chain) ---
    import numpy as _np
    for _c in chlorophylls: _c['is_red']=False
    _cx=_np.array([c['mg_coords'] for c in chlorophylls if c['chain_id'] in CORE_CHAIN_IDS and c.get('mg_coords') is not None])
    _ac={}
    for _c in chlorophylls:
        if _c['chain_id'] not in CORE_CHAIN_IDS and _c.get('mg_coords') is not None:
            _ac.setdefault(_c['chain_id'],[]).append(_c)
    _belt1=set()
    for _ch,_l in _ac.items():
        _p=_np.array([c['mg_coords'] for c in _l])
        if _np.min(_np.linalg.norm(_p[:,None,:]-_cx[None,:,:],axis=2))<30.0: _belt1.add(_ch)
    _nred=0
    for _ch in _belt1:
        _l=_ac[_ch]
        if len(_l)<2: continue
        _best=None
        for _a in range(len(_l)):
            for _b in range(_a+1,len(_l)):
                _d=_np.linalg.norm(_l[_a]['mg_coords']-_l[_b]['mg_coords'])
                if _best is None or _d<_best[0]: _best=(_d,_l[_a],_l[_b])
        if _best and _best[0]<11.0:
            _best[1]['is_red']=True; _best[2]['is_red']=True; _nred+=1
    if _ov.environ.get("BELT_FUNNEL", "0") == "1":
        print(f"--- belt-1 red dimers (exempt from the funnel): {_nred} in {len(_belt1)} chains ---")

    # -------------------------------

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
    _p7 = float(_o.environ.get("P700_E", "0"))
    if _p7 > 0:
        for _c in chlorophylls:
            if (_c["chain_id"], _c["res_seq"]) in RC_IDS: _c["site_energy"] = _p7
        print("--- P700 set to %.0f ---" % _p7)
    v_mn_global, h_ex_mean = calculate_hamiltonian(chlorophylls, stochastic=False)
    exciton_domains_indices = partition_into_exciton_domains(v_mn_global, h_ex_mean, V_CUTOFF, SITE_ENERGY_DIFF_CUTOFF)

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
    print('[P700-S] per-exciton S: 1.0 on P700 Chls, %.2f elsewhere' % HUANG_RHYS_S_ANT)
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
        all_redfield_rates = [calculate_redfield_rates(d, chlorophylls) for d in domain_results]

        all_abs_shapes = []
        all_ems_shapes = []

        # 3. Lineshapes
        for i, d in enumerate(domain_results):
            n_exc = len(d['local_energies'])
            d_abs = []; d_ems = []

            # Identify physics parameters for this domain
            is_red = any(chlorophylls[gx]['is_red'] for gx in d['global_indices'])
            is_p700 = any(chlorophylls[gx]['is_rc'] for gx in d['global_indices'])
            if is_p700:
                g_t_use = g_t_core
            elif is_red:
                g_t_use = g_t_red
            else:
                g_t_use = g_t_ant

            for alpha in range(n_exc):
                k_out = np.sum(all_redfield_rates[i][alpha, :])

                if k_out > 1e-10:
                    lifetime_est_ps = 1.0 / k_out
                else:
                    lifetime_est_ps = 10000.0

                energy_cm = renormalized_energies[i][alpha]
                gamma_mat = calculate_gamma(d, chlorophylls)
                gamma_aa = gamma_mat[alpha, alpha]
                g_t_use = _S_per_exciton(d, chlorophylls)[alpha] * _G_PH1 + _G_VIB

                abs_shape = calculate_optical_lineshape_full(omega_range_cm, energy_cm, gamma_aa, lifetime_est_ps, False, g_t_use, t_lineshape_axis)
                ems_shape = calculate_optical_lineshape_full(omega_range_cm, energy_cm, gamma_aa, lifetime_est_ps, True, g_t_use, t_lineshape_axis)
                d_abs.append(abs_shape); d_ems.append(ems_shape)
            all_abs_shapes.append(d_abs); all_ems_shapes.append(d_ems)

        # 4. Forster Rates
        all_forster_rates = {}
        for i in range(len(domain_results)):
            for j in range(len(domain_results)):
                if i == j: continue
                dist = get_min_domain_distance(domain_results[i], domain_results[j], chlorophylls)
                if dist < INTER_DOMAIN_DISTANCE_CUTOFF:
                    V_eff = calculate_interdomain_coupling(domain_results[i], domain_results[j], v_mn_global)
                    rates = calculate_generalized_forster_rates(V_eff, all_ems_shapes[i], all_abs_shapes[j], omega_range_cm)
                    all_forster_rates[(i, j)] = rates
        domain_rates_i = calculate_domain_level_rates(domain_results, all_forster_rates, rc_indices)
        sum_domain_rates += domain_rates_i

        # 5. Weighted Trapping Rates
        trapping_rates_i = calculate_weighted_trapping_rates(domain_results, chlorophylls, CHARGE_SEP_RATE_PS, TEMP_K)
        sum_trapping_rates += trapping_rates_i

    avg_rates = sum_domain_rates / N_MONTE_CARLO_RUNS
    avg_trapping_rates = sum_trapping_rates / N_MONTE_CARLO_RUNS

    print("\n--- Step 3: Final Simulation ---")
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
    pop_excited = history[:, :len(domain_info)].sum(axis=1)
    pop_yield = history[:, CHARGE_SEP_IDX]

    # --- Plotting ---
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    
    plt.figure(figsize=(8, 6))
    plt.plot(time_points, pop_excited, label='Excited population', linewidth=2)
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
    output_data = np.column_stack((time_points, pop_excited, pop_yield))
    np.savetxt("population_dynamics.txt", output_data,
               header="Time(ps)\tExcited_Pop\tYield",
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
