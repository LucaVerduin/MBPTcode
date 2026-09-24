"""Self-consistent GW (scGW): the dressed Green's function via Dyson.

`solve_qp_energy_space_time` only ever needs Sigma_c(i.omega) projected onto
one orbital diagonal, on a handful of Pade sample points -- enough to find
that orbital's QP pole, useless for a Dyson loop. Building an actual dressed
G(i.omega) needs the FULL Sigma_c(i.omega) matrix, on the SAME frequency grid
chi0/W already live on, so it round-trips cleanly to imaginary time for the
next iteration's polarizability. This module is deliberately kept apart from
space_time.py -- the O(N^3) route's tested, shared entry point -- rather than
threading a second output contract through it.

Sigma_c(i.omega) -> G(i.omega) here is the matrix analogue of the
W = [I - chi0]^-1 Dyson inversion `sigma_space_time` already does for the
screened interaction:

    G(i.omega) = [(i.omega + mu) I - diag(eps) - Sigma_c(i.omega)]^-1

Turning this into an actual self-consistency loop still needs G(i.omega)
transformed back to G(i.tau) and fed into a dressed replacement for
`chi0_imaginary_frequency` (which currently bakes in the bare-orbital
exponential form of G0) -- not implemented yet; this module stops at one
Dyson-dressed G.
"""
import numpy as np

from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.Base.utils.grids import (gauss_legendre_grid, minimax_time_grid,
                                  minimax_frequency_grid, minimax_supported_sizes)
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_transform_weights,
                                           COSINE_TW, COSINE_WT, SINE_TW, SINE_WT)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.imaginary_time import (self_energy_fit_ranges,
                                                   self_energy_matrix_imaginary_time,
                                                   sigma_ao_to_mo,
                                                   minimax_points_for_gw,
                                                   DEFAULT_TAU_TARGET)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency

DEFAULT_NTAU = 'auto'
DEFAULT_NFREQ = 'auto'

def get_freq_points(nfreq, ntau, e_min, e_max, w0):
    """ This function gets the frequency points either via the 
    minimax method, or via gauss-legendre method, just like
    in space_time.py
    """

    if nfreq is None or (isinstance(nfreq, str) and nfreq.lower() == 'auto'):
        if ntau not in minimax_supported_sizes():
            raise ValueError(
                f"nfreq='auto' needs a tabulated minimax frequency grid at "
                f'ntau = {ntau}; GreenX has {minimax_supported_sizes()}. Pass an '
                'explicit nfreq.')
        freq_points, freq_weights = minimax_frequency_grid(ntau, e_min, e_max)
    else:
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)

    return freq_points, freq_weights

def find_freq_points(nocc, eps, mu, ntau, e_min, e_max, w0, maxiter=5,
                     start_points=6, step=4, trace_tol=1e-8):
    """Sanity check that Tr[gamma] (built from the bare G0) actually
    converges as Gauss-Legendre points are added, THEN return the same
    minimax('auto') grid get_freq_points would have given anyway.

    Gauss-Legendre is only a cheap PROBE here, used because it can be built
    at any size (minimax cannot -- only the tabulated sizes in
    minimax_supported_sizes exist): a handful of GL sizes (maxiter, small by
    default) is enough to see whether the density integral is behaving, not
    to find a grid size to actually use. Regardless of the outcome, the
    frequency grid this function returns is unchanged: the minimax grid tied
    to ntau, exactly what get_freq_points('auto', ...) gives.
    """
    nmo = len(eps)
    occ_ref = np.zeros(nmo)
    occ_ref[:nocc] = 1.0

    n_freq = start_points
    freq_points_gl, freq_weights_gl = get_freq_points(n_freq, ntau, e_min, e_max, w0)
    sigma_0 = np.zeros((n_freq, nmo, nmo), dtype=complex)
    G0_gl = dyson_green_function(sigma_0, eps, mu, freq_points_gl)
    gamma_gl = density_matrix_scgw(G0_gl, freq_weights_gl)
    d_occ_old = np.max(np.abs(np.diag(gamma_gl) - occ_ref))

    converged = d_occ_old < trace_tol
    for _ in range(maxiter):
        if converged:
            break

        n_freq += step
        sigma_0 = np.zeros((n_freq, nmo, nmo), dtype=complex)
        freq_points_gl, freq_weights_gl = get_freq_points(n_freq, ntau, e_min, e_max, w0)
        G0_gl = dyson_green_function(sigma_0, eps, mu, freq_points_gl)
        gamma_gl = density_matrix_scgw(G0_gl, freq_weights_gl)
        d_occ = np.max(np.abs(np.diag(gamma_gl) - occ_ref))

        if d_occ_old - d_occ < 0:
            # stopped improving: not going to converge further this way
            break
        converged = d_occ < trace_tol
        d_occ_old = d_occ

    if converged:
        print(f'Gauss-Legendre density check converged: max|d_occ|={d_occ_old:.2e} '
             f'at nfreq={n_freq}')
    else:
        print(f'WARNING: Gauss-Legendre density check did NOT converge within '
             f'{maxiter} probes (max|d_occ|={d_occ_old:.2e} at nfreq={n_freq})')

    return get_freq_points('auto', ntau, e_min, e_max, w0)


def sigma_matrix_scgw_iteration1(mf, mol, nocc, ntau=DEFAULT_NTAU, nfreq=DEFAULT_NFREQ,
                      w0=1.0, auxbasis=None, radii=None, factors=None,
                      tau_target=DEFAULT_TAU_TARGET, timings=None):
    """Sigma_c(i.omega) as a full MO matrix, on chi0/W's own frequency grid.

    Same separable-factor/tau/frequency setup as `sigma_space_time`, but the
    self-energy is projected with `sigma_ao_to_mo` (full matrix) instead of
    the diagonal, and evaluated with `omega_out = freq_points` instead of a
    Pade grid -- the grid chi0/W already live on, since a Dyson loop needs
    Sigma_c and G comparable, frequency by frequency, to what built W.

    Returns (sigma_mo, eps, mu, freq_points, freq_weights); sigma_mo is
    (nfreq, nmo, nmo) complex. `freq_weights` are the matching minimax/Gauss-
    Legendre quadrature weights on `[0, infty)` -- not needed to build G
    itself (`dyson_green_function` closes the Dyson equation pointwise, no
    integral), but exactly what a frequency-integral quantity built from G,
    such as the density matrix, needs next.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, tau_err = minimax_points_for_gw(eps, nocc, mu=mu, target=tau_target)
        if timings is not None:
            timings['ntau_auto'] = ntau
            timings['tau_fit_error'] = tau_err

    X_mo, D, X_ao, coords = (factors if factors is not None
                             else separable_factors(mf, mol, auxbasis=auxbasis,
                                                    radii=radii))

    # find out if auto is better than just as many gauss-legendre points
    # (in reference to density trace)
    freq_points, freq_weights = find_freq_points(nocc, eps, mu, ntau, e_min, e_max, w0)

    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           freq_points, freq_weights)
    _, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    tau_points = 0.5 * minimax_time_grid(ntau, *rS)[0]

    print(f'cos: {grid.cosft_tw.shape}',f'sin: {grid.sinft_tw.shape}')

    chi0 = chi0_imaginary_frequency(X_mo, D, eps, nocc, grid, mu=mu)
    eye = np.eye(chi0.shape[-1])
    for k in range(chi0.shape[0]):
        chi0[k] = np.linalg.inv(eye - chi0[k])
    W_omega = chi0

    # omega_out = freq_points, not a Pade grid: Sigma comes back on the SAME
    # axis W lives on, so G built from it stays on that axis too.
    sigma_ao = self_energy_matrix_imaginary_time(
        X_ao, D, W_omega, mf.mo_coeff, eps, nocc,
        tau_points, freq_points, freq_points, mu=mu)
    sigma_mo = sigma_ao_to_mo(sigma_ao, mf.mo_coeff)

    return sigma_mo, eps, mu, freq_points, freq_weights


def dyson_green_function(sigma_mo, eps, mu, freq_points):
    """G(i.omega) = [(i.omega + mu) I - diag(eps) - Sigma(i.omega)]^-1.

    The matrix Dyson equation, closed pointwise per frequency point -- the
    same structure as the W = [I - chi0]^-1 inversion `sigma_space_time`
    already does for the screened interaction, with the analytic
    (i.omega + mu) I - diag(eps) standing in for G0^-1 (MO basis, so S = I
    and F = diag(eps)) instead of the identity chi0 is inverted against.

    sigma_mo: (nfreq, nmo, nmo) complex, on `freq_points`. Returns G, same
    shape.
    """
    nmo = len(eps)
    eye = np.eye(nmo)
    diag_eps = np.diag(eps)
    G = np.empty_like(sigma_mo)

    for k, w in enumerate(freq_points):
        g0_inv = (1j * w + mu) * eye - diag_eps
        G[k] = np.linalg.inv(g0_inv - sigma_mo[k])
    return G


def density_matrix_scgw(G, freq_weights):
    """gamma_pq = (1/2) delta_pq + (1/pi) sum_k w_k Re[G_pq(i.omega_k)].

    The T=0 density matrix from a Matsubara Green's function, as a frequency
    integral over the SAME [0, infty) grid G was built on -- `freq_weights`
    must be the quadrature paired with the `freq_points` `G` lives on
    (`sigma_matrix_scgw`'s third return value), not an independent choice.

    Exact for the bare G0: Re[G0_pp(i.omega)] = -(eps_p-mu)/(omega^2+(eps_p-mu)^2)
    integrates to +-1/2, so gamma_pp comes out exactly 1 (occupied) or 0
    (virtual) -- the mean-field occupations, recovered without ever touching
    eps directly. For the dressed G this is only as good as the quadrature:
    Tr(gamma) should reproduce the electron count, and that check is NOT
    automatic just because the same grid integrates chi0/W well.

    G: (nfreq, nmo, nmo) complex, on `freq_points`. Returns gamma, (nmo, nmo)
    real, in the MO basis G was built in (mf.mo_coeff @ gamma @ mf.mo_coeff.T
    for the AO density).
    """
    nmo = G.shape[-1]
    averaged_term = (1.0 / np.pi) * np.einsum('k,kpq->pq', freq_weights, G.real)

    # with np.printoptions(precision=5, suppress=True, linewidth=120):
    #     print('(1/pi) sum_k w_k Re[G(i.omega_k)]:')
    #     print(averaged_term + 0.5*np.eye(nmo))

    return 0.5 * np.eye(nmo) + averaged_term

def density_matrix_scgw_split(G, G_hf, F, mu, freq_weights):
    """Method for calculating the desity like
    Grumet, Liu, Kaltak, Klimes, Kresse, Phys. Rev. B 98 , 155143 (2018)

    Should be more accurate than density_matrix_scgw
    since the gamma_hf part is solved analytically, without quadrature error

    F: the Fock matrix G_hf was built from. gamma_hf is diagonal in F's OWN
    eigenbasis, not necessarily the caller's original orbital basis -- once
    F is rebuilt from a scGW density (mu_cycle), it is no longer diagonal
    there, so its eigenvalues/eigenvectors (not a fixed `eps`) are what
    decide occupation. Reduces exactly to the old eps-based version when
    F = diag(eps): eigh of a diagonal matrix returns eps itself, in order,
    with the identity as eigenvectors.
    """

    eps_F, U = np.linalg.eigh(F)
    occ_F = (eps_F < mu).astype(float)
    gamma_hf = U @ np.diag(occ_F) @ U.T

    G_c = G - G_hf

    gamma_c = (1/np.pi)*np.einsum('k,kpq->pq', freq_weights, G_c.real)

    # print(np.trace(gamma_c+gamma_hf))

    return gamma_c + gamma_hf

def build_new_F(gamma, V, h_mo):
    J_ij = np.einsum('ijkl,kl->ij', V, gamma)
    K_ij = np.einsum('ikjl,kl->ij', V, gamma)

    return h_mo + 2* J_ij - K_ij

def build_G_from_F(mu, freq_points, sigma_mo, F, return_G_hf=False):

    G = np.empty_like(sigma_mo)
    nmo = G.shape[-1]
    eye = np.eye(nmo)

    if not return_G_hf:
        for k, w in enumerate(freq_points):
            G[k] = np.linalg.inv((1j* w + mu)*eye - F - sigma_mo[k])
        return G
    
    else:
        G_hf = np.empty_like(sigma_mo)
        for k, w in enumerate(freq_points):
            G_hf_inv = (1j* w + mu)*eye - F 
            G[k] = np.linalg.inv(G_hf_inv - sigma_mo[k])
            G_hf[k] = np.linalg.inv((G_hf_inv))
        return G, G_hf

def _mu_cycle_find_trace(mu, freq_points, freq_weights, sigma_mo, gamma, V, h_mo,
                         densmethod='full'):
    """One (F, G, gamma) update at a trial mu. Returns (trace, gamma, G, F).

    Sigma (sigma_mo) is the only thing held fixed across a whole mu_cycle
    call; F is rebuilt from the CURRENT gamma estimate at every trial mu
    (the paper's inner loop over G, F, mu together, with only Sigma frozen
    until the next outer iteration) and threaded forward from one trial to
    the next by the caller.
    """
    F = build_new_F(gamma, V, h_mo)
    if densmethod.lower() == 'split':
        G, G_hf = build_G_from_F(mu, freq_points, sigma_mo, F, return_G_hf=True)
        gamma = density_matrix_scgw_split(G, G_hf, F, mu, freq_weights)
    else:
        G = build_G_from_F(mu, freq_points, sigma_mo, F)
        gamma = density_matrix_scgw(G, freq_weights)
    return np.trace(gamma), gamma, G, F

def mu_cycle(mu, G, gamma, nocc, freq_points, freq_weights, sigma_mo, V, h_mo,
            mu_step=0.1, trace_tol=1e-8, mu_tol=1e-10,
            max_bracket_iter=50, max_bisect_iter=100, densmethod='full'):
    """Bisection search for mu with Tr[gamma(mu)] = nocc.

    Sigma (sigma_mo) is held FIXED for the whole call; F and gamma are
    rebuilt from each other at every trial mu -- G, F, and mu form one inner
    loop, per the paper, threaded forward from one trial to the next rather
    than reset each time.

    Two phases: an expanding-step search to bracket the root (step grows
    while still hunting for the sign flip), then ordinary bisection, whose
    bracket width halves every iteration on its own.

    densmethod: passed straight through to _mu_cycle_find_trace at every
    trial mu -- 'split' builds G_hf alongside G and uses
    density_matrix_scgw_split; 'full' (default) uses the plain
    frequency-integral density_matrix_scgw.

    Returns (mu, G, gamma, F).
    """
    F = build_new_F(gamma, V, h_mo)
    trace_old = np.trace(gamma)

    print(f'Trace before mu-cycle: {trace_old}')

    if abs(trace_old - nocc) < trace_tol:
        return mu, G, gamma, F

    # --- Phase 1: bracket the root ---
    step = mu_step
    direction = -1.0 if trace_old > nocc else 1.0
    mu_a, trace_a, gamma_a = mu, trace_old, gamma

    for _ in range(max_bracket_iter):
        mu_b = mu_a + direction * step
        trace_b, gamma_b, G_b, F_b = _mu_cycle_find_trace(
            mu_b, freq_points, freq_weights, sigma_mo, gamma_a, V, h_mo,
            densmethod=densmethod)

        if (trace_a - nocc) * (trace_b - nocc) <= 0:
            break
        mu_a, trace_a, gamma_a = mu_b, trace_b, gamma_b
        step *= 1.5
    else:
        raise RuntimeError(f'mu_cycle: failed to bracket nocc={nocc} '
                           f'within {max_bracket_iter} probes')

    # --- Phase 2: bisection ---
    mu_lo, mu_hi, trace_lo, gamma_lo = mu_a, mu_b, trace_a, gamma_a
    for _ in range(max_bisect_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        trace_mid, gamma_mid, G_mid, F_mid = _mu_cycle_find_trace(
            mu_mid, freq_points, freq_weights, sigma_mo, gamma_lo, V, h_mo,
            densmethod=densmethod)

        if abs(trace_mid - nocc) < trace_tol or abs(mu_hi - mu_lo) < mu_tol:
            print(f'mu_cycle converged: mu={mu_mid}, trace={trace_mid}\n')
            return mu_mid, G_mid, gamma_mid, F_mid

        if (trace_lo - nocc) * (trace_mid - nocc) <= 0:
            mu_hi = mu_mid
        else:
            mu_lo, trace_lo, gamma_lo = mu_mid, trace_mid, gamma_mid

    raise RuntimeError(f'mu_cycle: did not converge in {max_bisect_iter} '
                       f'iterations (|trace-nocc| = {abs(trace_mid - nocc):.2e})')

def calc_new_sigma(G, freq_points, tau_points, F, mu, X_mo, D):
    """G(i.omega) -> the next iteration's Sigma_c(i.omega), via the dressed
    polarizability chi, the Dyson-inverted W, and Sigma = i G W as a
    pointwise product in imaginary time -- the same construction as
    chi0_imaginary_frequency/polarizability_imaginary_time and
    self_energy_matrix_imaginary_time, generalized from the bare-orbital G0
    to an arbitrary dressed G. Returns (G_tau, sigma_new); sigma_new is in
    MO basis, on the SAME freq_points G came in on, ready for the next
    build_G_from_F call.

    The cosine/sine transform weights are fit to F's OWN pole range, not
    chi0/Sigma's transition-energy range (e_min/e_max) -- reusing that range
    silently gives wrong results here (checked: ~0.29 vs ~0.0015 max error
    against the exact bare-G0 reference), since F is generally no longer
    diagonal in the original eps once it is rebuilt from a scGW density.

    Pi_PQ = Ghat_greater_PQ * Ghat_lesser_PQ (elementwise) is only meaningful
    in the real-space/ISDF (M, M) interpolation-point basis, NOT the MO
    basis G_lesser/G_greater come out of the transform in -- an elementwise
    product is not basis-covariant. X_mo projects each branch onto that
    basis first, exactly like the bare X_o/X_v collocation matrices do for
    the same purpose in polarizability_imaginary_time; D then projects the
    result on to the auxiliary basis.

    The prefactor here is +2.0, not the -2.0 polarizability_projected_tau
    uses on its Go*Gv: Gv = -G_greater (greens_function_imaginary_time puts
    an explicit minus sign on the virtual/greater branch that the bare
    Go*Gv product does not carry), so Go*Gv = G_lesser*(-G_greater), and
    that -1 flips polarizability_projected_tau's -2.0 into +2.0 here.
    Checked against chi0_imaginary_frequency in the bare-G0 limit: -2.0 was
    off by O(1) (not the ~1e-3 level the G-transform's own fit error would
    predict) and shrank the WRONG way with a more accurate ntau; +2.0 lines
    up with chi0_imaginary_frequency to 2e-3 at ntau=20 (the ntau=16 default
    here is right at the edge of accurate for this comparison specifically).
    """
    eps_F, _ = np.linalg.eigh(F)
    dG = np.abs(eps_F - mu)
    rG = (0.3 * dG.min(), 3.0 * dG.max())

    Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points, *rG, warn=False)
    Stw, _ = minimax_transform_weights(SINE_WT, tau_points, freq_points, *rG, warn=False)

    G_even = np.einsum('tk,kpq->tpq', Ctw, G.real)
    G_odd = np.einsum('tk,kpq->tpq', Stw, G.imag)

    G_lesser = G_even - G_odd
    G_greater = G_even + G_odd

    # MO basis -> ISDF/THC interpolation-point basis, per tau point.
    Ghat_lesser = np.einsum('pi,tij,qj->tpq', X_mo, G_lesser, X_mo, optimize=True)
    Ghat_greater = np.einsum('pi,tij,qj->tpq', X_mo, G_greater, X_mo, optimize=True)

    Pi_thc = Ghat_greater * Ghat_lesser

    # THC grid -> auxiliary (DF) basis. +2.0, not -2.0 -- see docstring.
    chi_tau = 2.0 * np.einsum('Pa,tPQ,Qb->tab', D, Pi_thc, D, optimize=True)

    # tau -> omega: chi is real/even, like chi0 (a two-particle/transition-
    # energy object), so only the cosine transform is needed, no sine half.
    # Fit against F's own TRANSITION range -- the occupied/virtual split by
    # mu among F's eigenvalues -- not G's per-orbital rG from above; this is
    # the same e_min/e_max chi0_imaginary_frequency itself needs, just
    # generalized off F instead of the stale bare eps.
    occ_mask = eps_F < mu
    e_min_F = eps_F[~occ_mask].min() - eps_F[occ_mask].max()
    e_max_F = eps_F[~occ_mask].max() - eps_F[occ_mask].min()
    Ctw_chi, _ = minimax_transform_weights(COSINE_TW, tau_points, freq_points,
                                           e_min_F, e_max_F, warn=False)
    chi_omega = np.einsum('wt,tab->wab', Ctw_chi, chi_tau, optimize=True)

    # Dyson invert chi(i.omega) -> W(i.omega), same pointwise pattern as the
    # chi0 -> W inversion elsewhere.
    naux = chi_omega.shape[-1]
    eye_aux = np.eye(naux)
    W_omega = np.empty_like(chi_omega)
    for k in range(chi_omega.shape[0]):
        W_omega[k] = np.linalg.inv(eye_aux - chi_omega[k])

    # omega -> tau: only the correlation part (W - I) decays and needs
    # transforming; W is real/even like chi, so cosine only. Padded, like
    # self_energy_fit_ranges' rW -- the screened interaction has weight
    # BELOW the smallest transition, so an unpadded range misfits exactly
    # where W is largest.
    rW_F = (0.3 * e_min_F, 3.0 * e_max_F)
    Ctw_W, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points, *rW_F, warn=False)
    Wt_tau = np.einsum('tk,kab->tab', Ctw_W, W_omega - eye_aux, optimize=True)

    # aux basis -> ISDF/THC grid, per tau point: Zt(tau) = D Wt(tau) D^T.
    Zt_tau = np.einsum('Pa,tab,Qb->tPQ', D, Wt_tau, D, optimize=True)

    # Sigma = i G W, pointwise product in imaginary time, THC grid. Unlike
    # the Pi step, NO extra sign flip is needed here: self_energy_matrix_
    # imaginary_time builds the raw (no-minus) virtual-branch product and
    # explicitly SUBTRACTS it into its greater accumulator, which is exactly
    # equivalent to using G_greater (with its own built-in minus) directly
    # with a plain +, as done here.
    sig_lesser_thc = Zt_tau * Ghat_lesser
    sig_greater_thc = Zt_tau * Ghat_greater

    # THC grid -> MO basis directly (X_mo for both indices -- Sigma comes
    # back in MO basis, no AO round trip needed, per self_energy_matrix_
    # imaginary_time's own documented X_ao/X_mo flexibility).
    sig_lesser = np.einsum('pi,tpq,qj->tij', X_mo, sig_lesser_thc, X_mo, optimize=True)
    sig_greater = np.einsum('pi,tpq,qj->tij', X_mo, sig_greater_thc, X_mo, optimize=True)

    even = sig_greater + sig_lesser
    odd = sig_greater - sig_lesser

    # tau -> omega: Sigma's own (padded) range, combining G's per-orbital
    # decay (dG) with W's transition range -- Sigma = G*W is a PRODUCT, so
    # its decay rates are SUMS of the two factors' own (self_energy_fit_
    # ranges' rS, generalized off F instead of the bare eps).
    rS_F = (0.3 * (dG.min() + e_min_F), 3.0 * (dG.max() + e_max_F))
    C, _ = minimax_transform_weights(COSINE_TW, tau_points, freq_points, *rS_F, warn=False)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, freq_points, *rS_F, warn=False)
    sigma_new = (np.einsum('wt,tij->wij', C, even, optimize=True)
                + 1j * np.einsum('wt,tij->wij', S, odd, optimize=True))

    return (G_lesser + 1j * G_greater), sigma_new

def solve_qp_energy_scgw(mf, mol, nocc, densmethod='split', **kwargs):
    """One scGW iteration: the dressed Green's function via Dyson.

    In the future going to run the sc cycle

    `sigma_matrix_scgw` builds Sigma_c(i.omega) as a full matrix on chi0/W's
    own frequency grid; this closes the Dyson equation pointwise per
    frequency. See the module docstring for what is still missing to turn
    this into an actual self-consistency loop (the i.omega -> i.tau
    transform of G and a dressed replacement for `chi0_imaginary_frequency`).

    Returns (G, sigma_mo, eps, mu, freq_points, freq_weights, F_new). F_new is
    h_mo + J[gamma] - 0.5*K[gamma] (see `build_new_F`), the updated Fock
    matrix in mf.mo_coeff's basis built from the scGW density `gamma`.
    """
    sigma_mo, eps, mu, freq_points, freq_weights = sigma_matrix_scgw_iteration1(
        mf, mol, nocc, **kwargs)

    G = dyson_green_function(sigma_mo, eps, mu, freq_points)

    if densmethod.lower()=='split':
        G0 = dyson_green_function(np.zeros_like(sigma_mo), eps, mu, freq_points)
        gamma = density_matrix_scgw_split(G, G0, np.diag(eps), mu, freq_weights)

    elif densmethod.lower()=='full':
        gamma = density_matrix_scgw(G, freq_weights)

    else:
        raise KeyError('densemethod implemented "full" or "split"')

    #here
    print('\nDIAGNOSTICS\n')
    h_mo = mf.mo_coeff.T @ mf.get_hcore(mol) @ mf.mo_coeff
    V = get_two_electron_integrals_chemist(mol,mf)

    # print(f'Testing mu-cycle: mu:{mu}, tr_gamma:{np.trace(gamma)}, expected 1')
    mu, G, gamma, F = mu_cycle(mu, G, gamma, nocc, freq_points,
                               freq_weights, sigma_mo, V, h_mo, densmethod=densmethod)

    # Evaluate now density of G
    n_occ_gamma = np.trace(gamma)
    print(f'n_occ_gamma {n_occ_gamma}, expected 1.0')

    return G, sigma_mo, eps, mu, freq_points, freq_weights, F
