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

from src.Base.pyscf_interface import get_orbital_energies
from src.Base.utils.grids import (gauss_legendre_grid, minimax_time_grid,
                                  minimax_frequency_grid, minimax_supported_sizes)
from src.Base.utils.time_frequency import TimeFrequencyGrid
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

def find_freq_points(nocc, eps, mu, ntau, e_min, e_max, w0):
    """This function compares the trace of density of the noninteracting 
    greens function to what is expected and uses that to determine
    if minimax or gauss-legendre should be used.
    """

    nmo = len(eps)

    freq_points_mm, freq_weights_mm = get_freq_points('auto', ntau, e_min, e_max, w0)
    n_freq_mm = len(freq_points_mm)
    freq_points_gl, freq_weights_gl = get_freq_points(n_freq_mm, ntau, e_min, e_max, w0)

    sigma_0 = np.zeros((n_freq_mm, nmo, nmo), dtype=complex)

    G0_mm = dyson_green_function(sigma_0, eps, mu, freq_points_mm)
    G0_gl = dyson_green_function(sigma_0, eps, mu, freq_points_gl)

    gamma_mm = density_matrix_scgw(G0_mm, freq_weights_mm)
    gamma_gl = density_matrix_scgw(G0_gl, freq_weights_gl)

    occ_ref = np.zeros(len(eps))
    occ_ref[:nocc] = 1.0

    d_occ_mm = np.max(np.abs(np.diag(gamma_mm) - occ_ref))
    d_occ_gl = np.max(np.abs(np.diag(gamma_gl) - occ_ref))

    if d_occ_mm < d_occ_gl:
        print('using mm grid')
        return freq_points_mm, freq_weights_mm
    else:
        print('using gl grid')
        return freq_points_gl, freq_weights_gl


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
    nmo = len(eps)
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

def density_matrix_scgw_split(G, G0, eps, mu, freq_weights):
    """Method for calculating the desity like 
    Grumet, Liu, Kaltak, Klimes, Kresse, Phys. Rev. B 98 , 155143 (2018)

    Should be more accurate than density_matrix_scgw 
    since the gamma_hf part is solved analytically, without quadrature error
    """

    nmo = G.shape[-1]
    gamma_hf = np.zeros((nmo,nmo))
    gamma_c  = np.zeros((nmo,nmo))

    for i, epsilon in enumerate(eps):
        if mu-epsilon > 0:
            gamma_hf[i,i] = 1

    G_c = G - G0

    gamma_c = (1/np.pi)*np.einsum('k,kpq->pq', freq_weights, G_c.real)

    # print(np.trace(gamma_c+gamma_hf))

    return gamma_c + gamma_hf


def solve_qp_energy_scgw(mf, mol, nocc, densmethod='split', **kwargs):
    """One scGW iteration: the dressed Green's function via Dyson.

    In the future going to run the sc cycle

    `sigma_matrix_scgw` builds Sigma_c(i.omega) as a full matrix on chi0/W's
    own frequency grid; this closes the Dyson equation pointwise per
    frequency. See the module docstring for what is still missing to turn
    this into an actual self-consistency loop (the i.omega -> i.tau
    transform of G and a dressed replacement for `chi0_imaginary_frequency`).

    Returns (G, sigma_mo, eps, mu, freq_points, freq_weights).
    """
    sigma_mo, eps, mu, freq_points, freq_weights = sigma_matrix_scgw_iteration1(
        mf, mol, nocc, **kwargs)

    G = dyson_green_function(sigma_mo, eps, mu, freq_points)

    if densmethod.lower()=='split':
        G0 = dyson_green_function(np.zeros_like(sigma_mo), eps, mu, freq_points)
        gamma = density_matrix_scgw_split(G, G0, eps, mu, freq_weights)

    elif densmethod.lower()=='full':
        gamma = density_matrix_scgw(G, freq_weights)

    else:
        raise KeyError('densemethod implemented "full" or "split"')

    return G, sigma_mo, eps, mu, freq_points, freq_weights
