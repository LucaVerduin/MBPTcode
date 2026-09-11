"""Space-time GW: the O(N^3) route to a quasiparticle energy.

The polarizability is built in imaginary time from a separable (ISDF)
factorization of the ERIs, where the occupied and virtual sums decouple, and
the self-energy Sigma = -G Wt is a pointwise product there rather than a
convolution.

Peer of the Casida route (`qp_energy.calc_qp_energy`) and the
imaginary-frequency route (`imaginary_axis.solve_qp_energy_imaginary_axis`);
all three end in `qp_solve.solve_qp_from_imaginary_axis`.

Four silent traps, all handled here:

  * W must come from the SAME factors as Sigma. pyscf's cderi is L^-1-whitened,
    a separable RI fits with the symmetric V^-1/2, and mixing the two gauges
    moves the QP energy by ~1.4 eV while staying self-consistent.
  * The tau grid uses the SELF-ENERGY's energy range, not the polarizability's:
    Sigma is a product, so its decay rates are sums.
  * The tau and frequency axes are decoupled; one grid cannot serve both the
    tau->omega transform and the Sigma quadrature.
  * Occupied states sample the negative branch, so the Pade input is conjugated.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- the space-time
method: Sigma = i G W as a pointwise product in imaginary time rather than a
convolution in frequency, which is what makes this route cubic.
Duchemin and Blase, J. Chem. Theory Comput. 17, 2383 (2021) -- the same
construction on a separable RI in a Gaussian basis, which this follows.
Foerster and Visscher, J. Chem. Theory Comput. 16, 7381 (2020) -- low-scaling
G0W0 in a localized basis, the pair-fitting ancestor of this route.
"""
import os
import time as _time

import numpy as np
from pyscf import df as pyscf_df

from src.Base.pyscf_interface import get_orbital_energies
from src.Base.solvent_screening import get_solvent_screening
from src.Base.separable_ri import (DEFAULT_PAIR_TOL, build_separable_ri,
                                   molecular_points_covariant,
                                   optimize_atomic_radii, published_grids)
from src.Base.utils.grids import (gauss_legendre_grid, minimax_time_grid,
                                  minimax_frequency_grid,
                                  minimax_supported_sizes)
from src.Base.utils.mpi_grid import grid_comm, partition, reduce_sum
from src.Base.utils.time_frequency import (TimeFrequencyGrid, COSINE_WT,
                                           minimax_transform_weights)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.imaginary_time import (self_energy_matrix_imaginary_time,
                                                   sigma_ao_to_mo_diagonal,
                                                   self_energy_fit_ranges,
                                                   screened_interaction_tau_blocked,
                                                   minimax_points_for_gw,
                                                   DEFAULT_TAU_TARGET)
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             solve_qp_from_imaginary_axis,
                                             imaginary_axis_sample_points)
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency

DEFAULT_NTAU = 'auto'
DEFAULT_NFREQ = 'auto'
DEFAULT_NPADE = 16
DEFAULT_COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}


def separable_factors(mf, mol, auxbasis=None, radii=None, counts=None,
                      block_memory_gb=4.0, pair_tol=DEFAULT_PAIR_TOL,
                      n_start=1):
    """(X_mo, D, X_ao, coords) of the Duchemin-Blase separable RI, Z = D D^T.

    X_ao is the collocation the fit actually produces; X_mo = X_ao C is the
    form most consumers want. Both are returned because inverting one back to
    the other needs a square C, which a large basis does not guarantee.

    Interpolation points come from covariant atomic frames, so the answer does
    not depend on the orientation of the molecule.

    radii:           per-element shell radii; optimized per element if omitted.
    block_memory_gb: caps the working set of the fit's blocked loops. It does
                     not change the answer, but it IS a speed knob as well as a
                     memory one -- one `aux_e2` call per block, each rebuilding
                     a shell-pair list over nbas x auxnbas. See `build_D_F`,
                     which carries the measurement; size it from the node.
    """
    auxbasis = auxbasis or (str(mol.basis) + '-ri')
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    counts = counts or DEFAULT_COUNTS

    if radii is None:
        pub = published_grids()
        radii, origins = {}, {}
        for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
            if el in pub and str(mol.basis).lower() == 'cc-pvtz':
                radii[el], origins[el] = pub[el]
            else:
                radii[el] = optimize_atomic_radii(el, mol.basis, auxbasis,
                                                  counts=counts,
                                                  n_start=n_start)[0]
                origins[el] = False
    else:
        origins = {el: False for el in radii}

    coords = molecular_points_covariant(mol, radii, origin_by_element=origins)
    X, _, M = build_separable_ri(mol, coords, auxmol=auxmol,
                                 block_memory_gb=block_memory_gb,
                                 pair_tol=pair_tol)

    # Symmetric V^1/2 gauge on the auxiliary index -- the one Sigma expects.
    V = auxmol.intor('int2c2e', aosym='s1')
    # An attached solvent screening substitutes v -> v + vtilde. The
    # interaction enters this factorization only through the auxiliary
    # metric (Z = D D^T fits pair densities against V), so dressing V is the
    # whole substitution -- the DF analogue is
    # SolventScreening.whitened_transform.
    screening = get_solvent_screening(mf)
    if screening is not None:
        V = V + screening.aux_kernel(auxmol)
    w, v = np.linalg.eigh(V)
    if screening is not None and w.min() < -1e-10 * w.max():
        raise RuntimeError(
            f"the screened auxiliary metric v + vtilde is indefinite "
            f"(smallest eigenvalue {w.min():.3e}): the reaction field "
            f"over-screens the bare interaction. Check eps and the cavity.")
    keep = w > 1e-12 * w.max()
    V_half = (v[:, keep] * np.sqrt(w[keep])) @ v[:, keep].T
    return X @ mf.mo_coeff, M.T @ V_half, X, coords


def _unpack_factors(factors):
    """(X_mo, D, X_ao, coords); the last two are None for a shorter tuple."""
    return tuple(factors) + (None,) * (4 - len(factors))


def _ao_collocation(X_mo, mf):
    """X_ao[k, mu] = chi_mu(r_k), inverted from the MO collocation X_mo = X_ao C.

    FALLBACK ONLY -- `separable_factors` now returns X_ao directly, because the
    inversion is not always available. It is exact where it works: MO
    coefficients are S-orthonormal, C^T S C = I, so C^-1 = C^T S. But it needs a
    square C, and a mean field that dropped linear dependencies gives only a
    left inverse, i.e. a silent projection. cc-pVQZ on the 178-atom
    chlorophyllide dimer is past that line -- cond(S) = 1.8e7 with four
    eigenvalues below 1e-6 -- so the AO route there must take X_ao from the fit.
    """
    C = mf.mo_coeff
    if C.shape[0] != C.shape[1]:
        raise ValueError(
            f'mo_coeff is {C.shape[0]}x{C.shape[1]}: the mean field dropped '
            f'{C.shape[0] - C.shape[1]} linearly dependent combinations, so the '
            'MO collocation cannot be inverted back to AO exactly. Build the '
            'factors in the AO representation instead.')
    return X_mo @ C.T @ mf.get_ovlp()


def _sigma_mo_diagonal(X_mo, D, W_omega, mf, eps, nocc, tau_points, freq_points,
                       pade_freq, mu, p_state, Wt_tau=None, tau_indices=None,
                       reduce_over=None, X_ao=None, coords=None,
                       screen_r_cut=None):
    """Sigma_c(i.omega) built in the AO basis, projected onto the p_state diagonal.

    Returns (nstates, nfreq), or a bare (nfreq,) for a scalar state.
    """
    if X_ao is None:
        X_ao = _ao_collocation(X_mo, mf)
    sigma_ao = self_energy_matrix_imaginary_time(
        X_ao, D, W_omega, mf.mo_coeff, eps, nocc,
        tau_points, freq_points, pade_freq, mu=mu, Wt_tau=Wt_tau,
        tau_indices=tau_indices, coords=coords, screen_r_cut=screen_r_cut)
    if reduce_over is not None:
        reduce_sum(sigma_ao, reduce_over)
    sigma = sigma_ao_to_mo_diagonal(sigma_ao, mf.mo_coeff,
                                    states=np.atleast_1d(p_state)).T
    del sigma_ao
    return sigma[0] if np.ndim(p_state) == 0 else sigma


def _finish_qp(sigma, eps, nocc, p_state, mu, pade_freq, mf, mol,
               solver_mode, dm_correction, greedy, timings, sigma_x='mf'):
    """Sigma_c on the imaginary axis -> quasiparticle energy, one per state."""
    states = np.atleast_1d(p_state)
    scalar = np.ndim(p_state) == 0
    sig = np.atleast_2d(sigma)

    _t = _time.time()
    # One exchange build for the whole window: <Sigma_x - v_xc> carries no state
    # index until it is indexed.
    xc_diag = static_exchange_diagonal(mf, mol, states,
                                       dm_correction=dm_correction,
                                       exchange=sigma_x)
    out = []
    for i, p in enumerate(states):
        z_fit, _ = imaginary_axis_sample_points(pade_freq, nocc, p, mu)
        # Occupied states sit on the negative branch, where Sigma(-i w) = conj.
        s_p = np.conj(sig[i]) if p < nocc else sig[i]
        out.append(solve_qp_from_imaginary_axis(eps, int(p), xc_diag[i],
                                                z_fit, s_p, greedy=greedy,
                                                solver_mode=solver_mode))
    if timings is not None:
        timings['t_qp'] = _time.time() - _t
    return out[0] if scalar else np.asarray(out)


def sigma_space_time(mf, mol, nocc, p_state,
                     ntau=DEFAULT_NTAU, nfreq=DEFAULT_NFREQ,
                     npade=DEFAULT_NPADE, w0=1.0, auxbasis=None,
                     radii=None, factors=None, timings=None,
                     distribute=False, comm=None,
                     freq_block=None, scratch_dir=None,
                     tau_target=DEFAULT_TAU_TARGET, extras=None,
                     screen_r_cut=None):
    """Sigma_c(i.omega) on the p_state diagonal, the space-time route.

    Everything `solve_qp_energy_space_time` does short of the Dyson QP solve:
    builds the separable factors, the tau/frequency grids, W(i.omega) or
    Wt(i.tau), and Sigma_c. Returns (sigma, eps, nocc, mu, pade_freq) -- what
    `_finish_qp` needs to close the Dyson equation per state.

    Split out from `solve_qp_energy_space_time` for a self-consistent (scGW)
    loop: an eigenvalue-only iteration re-runs exactly this with `eps` (hence
    `mu` and the tau/frequency grids) taken from the previous iteration's QP
    energies rather than the mean field's, while the root-finding step
    (`_finish_qp`) is shared and unchanged. `solve_qp_energy_space_time` is
    this function followed by that one step.

    See `solve_qp_energy_space_time` for the argument descriptions.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    # R = e_max/e_min grows as the gap closes, so a fixed ntau is wrong at one
    # end of any size series.
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, tau_err = minimax_points_for_gw(eps, nocc, mu=mu,
                                              target=tau_target)
        if timings is not None:
            timings['ntau_auto'] = ntau
            timings['tau_fit_error'] = tau_err

    X_mo, D, X_ao, coords = _unpack_factors(
        factors if factors is not None
        else separable_factors(mf, mol, auxbasis=auxbasis, radii=radii))

    # The frequency axis paired with the time axis: same size, minimax. It is
    # not a quadrature here -- chi0 is transformed onto it only so the Dyson
    # inversion, which is not diagonal in time, can be done, and W comes
    # straight back. A round trip on ntau points carries no more information.
    if nfreq is None or (isinstance(nfreq, str) and nfreq.lower() == 'auto'):
        if ntau not in minimax_supported_sizes():
            raise ValueError(
                f"nfreq='auto' needs a tabulated minimax frequency grid at "
                f'ntau = {ntau}; GreenX has {minimax_supported_sizes()}. Pass an '
                'explicit nfreq.')
        freq_points, freq_weights = minimax_frequency_grid(ntau, e_min, e_max)
    else:
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    pade_freq = gauss_legendre_grid(npade, w0=w0)[0]

    # Carry omega = 0 as a zero-weight passenger when the caller wants the
    # static screening, so a BSE need not repeat the tau sweep. Stripped again
    # below: omega = 0 is not part of the Sigma quadrature.
    want_static = extras is not None
    if want_static:
        freq_points = np.append(np.asarray(freq_points, float), 0.0)
        freq_weights = np.append(np.asarray(freq_weights, float), 0.0)

    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           freq_points, freq_weights)
    mpi_comm, rank, nranks = (grid_comm(comm) if distribute else (None, 0, 1))
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None

    _, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    tau_points = 0.5 * minimax_time_grid(ntau, *rS)[0]

    if freq_block or scratch_dir:
        sigma = _sigma_blocked(X_mo, D, mf, mol, eps, nocc, mu, grid, tau_points,
                               freq_points, pade_freq, p_state, want_static,
                               extras, ntau, nranks, freq_block, scratch_dir,
                               timings, X_ao, coords, screen_r_cut)
        return sigma, eps, nocc, mu, pade_freq

    _t = _time.time()
    chi0 = chi0_imaginary_frequency(X_mo, D, eps, nocc, grid, mu=mu,
                                    tau_indices=tau_mine)
    if nranks > 1:
        reduce_sum(chi0, mpi_comm)
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['nranks'] = nranks

    # Dyson, in place one frequency at a time: a list comprehension would hold
    # both the list and the stacked copy on top of chi0.
    _t = _time.time()
    eye = np.eye(chi0.shape[-1])
    for k in range(chi0.shape[0]):
        chi0[k] = np.linalg.inv(eye - chi0[k])
    W_omega = chi0
    if want_static:
        extras['w_static'] = W_omega[-1].copy()
        extras['w_static_ntau'] = ntau
        W_omega = W_omega[:-1]
        freq_points = freq_points[:-1]
        freq_weights = freq_weights[:-1]
    if timings is not None:
        timings['t_dyson'] = _time.time() - _t

    _t = _time.time()
    tau_mine_sig = partition(len(tau_points), rank, nranks) if nranks > 1 else None
    sigma = _sigma_mo_diagonal(X_mo, D, W_omega, mf, eps, nocc, tau_points,
                               freq_points, pade_freq, mu, p_state,
                               tau_indices=tau_mine_sig, X_ao=X_ao,
                               coords=coords, screen_r_cut=screen_r_cut,
                               reduce_over=mpi_comm if nranks > 1 else None)
    if timings is not None:
        timings['t_sigma'] = _time.time() - _t

    return sigma, eps, nocc, mu, pade_freq


def solve_qp_energy_space_time(mf, mol, nocc, p_state,
                               ntau=DEFAULT_NTAU, nfreq=DEFAULT_NFREQ,
                               npade=DEFAULT_NPADE, w0=1.0, auxbasis=None,
                               radii=None, factors=None, greedy=True,
                               solver_mode='pole_strength', dm_correction=None,
                               timings=None, distribute=False, comm=None,
                               freq_block=None, scratch_dir=None,
                               tau_target=DEFAULT_TAU_TARGET, extras=None,
                               screen_r_cut=None, sigma_x='mf'):
    """GW@RPA quasiparticle energy by the space-time route; restricted, DF only.

    Same quantity as `calc_qp_energy(selfenergy='GW', polarizability='RPA')`.
    `p_state` is one orbital or a window; a window shares a single Sigma.

    Sigma is built in the AO basis and projected onto the requested MO diagonal.
    That contraction is flat in the number of states and leaves the AO matrix
    behind for evGW/qsGW, at (npade, nao, nao) complex of memory.

    ntau:        imaginary-time points, or 'auto' to size from the
                 Kaltak-Klimes-Kresse test integral to residual `tau_target`.
    factors:     pre-built (X_mo, D), to reuse the factorization across calls.
    freq_block:  build W blockwise in frequency, so chi0 is never formed.
    scratch_dir: additionally cache the tau projection on disk.
    distribute:  split the tau sums over MPI ranks and all-reduce; exact, no
                 halo. The Dyson inversion stays replicated.
    extras:      dict; receives W(omega=0) for a BSE on the same factors.
    sigma_x:     which K builds the static exchange, see
                 `static_exchange_diagonal`. 'mf' on an ISDF mean field puts
                 the grid's K error into every QP energy at first order;
                 'df-direct' removes it with one streamed three-index pass and
                 no stored tensor, 'exact' with a full direct K.
    """
    sigma, eps, nocc, mu, pade_freq = sigma_space_time(
        mf, mol, nocc, p_state, ntau=ntau, nfreq=nfreq, npade=npade, w0=w0,
        auxbasis=auxbasis, radii=radii, factors=factors, timings=timings,
        distribute=distribute, comm=comm, freq_block=freq_block,
        scratch_dir=scratch_dir, tau_target=tau_target, extras=extras,
        screen_r_cut=screen_r_cut)
    return _finish_qp(sigma, eps, nocc, p_state, mu, pade_freq, mf, mol,
                      solver_mode, dm_correction, greedy, timings, sigma_x)


def _sigma_blocked(X_mo, D, mf, mol, eps, nocc, mu, grid, tau_points, freq_points,
                   pade_freq, p_state, want_static, extras, ntau, nranks,
                   freq_block, scratch_dir, timings, X_ao, coords, screen_r_cut):
    """Low-memory branch: chi0 is never formed. Returns Sigma_c(i.omega) on p_state.

    Frequencies are built, inverted and folded into Wt(i.tau) a block at a time,
    so the peak is Wt plus one block instead of the whole frequency axis.
    """
    # The blocked sweep owns the tau loop, so MPI tau-distribution cannot apply.
    if nranks > 1:
        raise ValueError('freq_block/scratch_dir cannot be combined with '
                         'distribute=True; the blocked sweep owns the tau loop.')
    rW, _ = self_energy_fit_ranges(eps, nocc, mu=mu)

    # Fit the omega -> tau weights on the UNEXTENDED axis, then pad with a zero
    # column: every output tau is fitted from all input frequencies, so letting
    # omega = 0 into the fit refits every other coefficient.
    static_idx = None
    fit_freqs = freq_points[:-1] if want_static else freq_points
    Ctw, w_fit_err = minimax_transform_weights(COSINE_WT, tau_points,
                                               fit_freqs, *rW, warn=False)
    if timings is not None:
        timings['w_fit_error'] = w_fit_err
    if want_static:
        static_idx = len(freq_points) - 1
        Ctw = np.hstack([Ctw, np.zeros((Ctw.shape[0], 1))])

    wt_path = os.path.join(scratch_dir, 'wt_tau.npy') if scratch_dir else None
    _t = _time.time()
    Wt_tau = screened_interaction_tau_blocked(
        X_mo, D, eps, nocc, grid, Ctw, mu=mu, freq_block=freq_block,
        scratch_dir=scratch_dir, wt_scratch=wt_path,
        static_index=static_idx, static_out=extras)
    if want_static:
        extras['w_static_ntau'] = ntau
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['t_dyson'] = 0.0
        timings['nranks'] = 1

    _t = _time.time()
    sigma = _sigma_mo_diagonal(X_mo, D, None, mf, eps, nocc, tau_points,
                               freq_points, pade_freq, mu, p_state,
                               Wt_tau=Wt_tau, X_ao=X_ao, coords=coords,
                               screen_r_cut=screen_r_cut)
    if timings is not None:
        timings['t_sigma'] = _time.time() - _t

    del Wt_tau
    if wt_path and os.path.exists(wt_path):
        os.remove(wt_path)
    return sigma


def solve_qp_diagonal_space_time(mf, mol, nocc, states=None, **kwargs):
    """The whole QP diagonal from one self-energy, as a BSE@GW needs it.

    The per-tau work Zt = D Wt D^T is shared by every state, so n states cost
    one extra (M, M) x (M, n) product.

    states: which orbitals, default all. Returns (qp_energies, states), Hartree.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    if states is None:
        states = np.arange(len(eps))
    states = np.atleast_1d(states)
    return solve_qp_energy_space_time(mf, mol, nocc, states, **kwargs), states
