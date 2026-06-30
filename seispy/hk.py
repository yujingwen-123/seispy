import numpy as np
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from obspy.io.sac.sactrace import SACTrace
import matplotlib.pyplot as plt
from os.path import join
from seispy.rfcorrect import RFStation
from seispy.hkpara import hkpara, HKPara
from seispy.geo import srad2skm
import argparse
from seispy.utils import load_cyan_map, array_instance


def transarray(array, axis=0):
    if not array_instance(array):
        raise ValueError('array should be `numpy.ndarray`')
    if len(array.shape) != 1:
        raise ValueError('array should be 1-d array')
    if axis == 0:
        return array.reshape(-1, array.shape[0])
    elif axis == 1:
        return array.reshape(array.shape[0], -1)
    else:
        raise ValueError('axis should be 0 or 1')


def vslow(v, rayp):
    return np.sqrt(1/(v**2) - rayp**2)


def tps(depth, eta_p, eta_s):
    return np.dot(transarray(eta_s - eta_p, axis=1), transarray(depth, axis=0))


def tppps(depth, eta_p, eta_s):
    return np.dot(transarray(eta_s + eta_p, axis=1), transarray(depth, axis=0))


def tpsps(depth, eta_s):
    return np.dot(transarray(2 * eta_s, axis=1), transarray(depth, axis=0))


def time2idx(times, ti0, dt):
    ti = ti0 + np.around(times / dt)
    return ti.reshape(ti.size).astype(int)


def hkstack(seis, t0, dt, p, h, kappa, vp=6.3, weight=(0.7, 0.2, 0.1),
            stack_method='linear', nth_order=4):
    """H-k stacking with multiple stacking methods.

    Parameters
    ----------
    seis : np.ndarray
        Receiver function data matrix (nrf x nt) or (nt x nrf)
    t0 : float
        Time of direct P arrival
    dt : float
        Sampling interval
    p : np.ndarray
        Ray parameters for each RF
    h : np.ndarray
        Array of H (Moho depth) values
    kappa : np.ndarray
        Array of Vp/Vs values
    vp : float
        P-wave velocity in km/s
    weight : tuple
        Weights for (0p1s, 2p1s, 1p2s) phases
    stack_method : str
        Stacking method:
        - 'linear': Standard linear stacking (Zhu & Kanamori, 2000)
        - 'nth_root': Nth-root stacking for noise suppression (Niu et al., 2007, eq. 2-3)
        - 'cc': Cross-correlation weighted stacking (Niu et al., 2007, eq. 4)
    nth_order : int
        Order of nth-root stacking, default 4 (as in Niu et al., 2007)

    Returns
    -------
    stack : np.ndarray
        Stacked amplitudes for each phase (nk, nh, 3)
    stackvar : np.ndarray
        Variance of stack
    Normed_stack : np.ndarray
        Normalized stacked energy (nk, nh)
    allstackvar : np.ndarray
        Variance of allstack
    """
    # get dimensions
    nh = len(h)
    nk = len(kappa)
    nrf = len(p)

    # check the orientation of the seis array
    if seis.shape[0] != nrf:
        seis = seis.T
        if seis.shape[0] != nrf:
            raise IndexError('SEIS array dimensions should be (nt x nrf)')

    # amp correction for Ps
    am_cor = 151.5478 * p ** 2 + 3.2896 * p + 0.2618

    # get all vs, single column
    vs = vp / kappa

    # get index of direct P
    ti0 = round(t0 / dt)

    # initialize stacks
    tstack = np.zeros((nk, nh, 3))

    use_nth_root = 'nth_root' in stack_method

    if use_nth_root:
        # For nth-root stacking: accumulate sign(x)*|x|^(1/N)
        nroot_stack = np.zeros((nk, nh, 3))
        stack2 = np.zeros((nk, nh, 3))
        allstack = np.zeros((nk, nh, nrf))
    else:
        stack = np.zeros((nk, nh, 3))
        stack2 = np.zeros((nk, nh, 3))
        allstack = np.zeros((nk, nh, nrf))

    for i in range(nrf):
        eta_p = vslow(vp, p[i])
        eta_s = vslow(vs, p[i])

        # get times of Ps for all combinations of vs and H
        t1 = time2idx(tps(h, eta_p, eta_s), ti0, dt)
        t2 = time2idx(tppps(h, eta_p, eta_s), ti0, dt)
        t3 = time2idx(tpsps(h, eta_s), ti0, dt)

        tstack[:, :, 0] = am_cor[i] * seis[i, t1].reshape(nk, nh)
        tstack[:, :, 1] = am_cor[i] * seis[i, t2].reshape(nk, nh)
        tstack[:, :, 2] = -am_cor[i] * seis[i, t3].reshape(nk, nh)

        if use_nth_root:
            # Nth-root stacking: sign(x) * |x|^(1/N) accumulated across events
            for j in range(3):
                nroot_stack[:, :, j] += (
                    np.sign(tstack[:, :, j]) * np.abs(tstack[:, :, j]) ** (1.0 / nth_order)
                )
            # Per-event weighted combination (for allstack variance later)
            allstack[:, :, i] = (
                weight[0] * tstack[:, :, 0] +
                weight[1] * tstack[:, :, 1] +
                weight[2] * tstack[:, :, 2]
            )
        else:
            stack += tstack
            stack2 += tstack ** 2
            allstack[:, :, i] = (
                weight[0] * tstack[:, :, 0] +
                weight[1] * tstack[:, :, 1] +
                weight[2] * tstack[:, :, 2]
            )

    if use_nth_root:
        # Convert nth-root accumulation to final stack: r * |r|^(N-1)
        r = nroot_stack / nrf
        stack = r * np.abs(r) ** (nth_order - 1)
        # Variance is not well-defined for nth-root stacking
        stackvar = np.zeros_like(stack)
        allstackvar = np.var(allstack, axis=2)
        allstack = np.mean(allstack, axis=2)
    else:
        stack = stack / nrf
        stackvar = (stack2 - stack ** 2) / (nrf ** 2)
        allstackvar = np.var(allstack, axis=2)
        allstack = np.mean(allstack, axis=2)

    # Cross-correlation weighted stacking (Niu et al., 2007, eq. 4)
    # Applies to both 'cc' and 'nth_root_cc' methods
    if 'cc' in stack_method:
        cc_weight = _compute_cc_weight(stack, nk)
        # Apply cross-correlation weight to each kappa slice
        allstack = allstack * cc_weight[:, np.newaxis]

    # Normalize
    Normed_stack = allstack - np.min(allstack)
    Normed_stack = Normed_stack / np.max(Normed_stack)
    return stack, stackvar, Normed_stack, allstackvar


def _compute_cc_weight(stack, nk):
    """Compute cross-correlation weight between the three phase stacks
    for each kappa (Vp/Vs ratio).

    Following Niu et al. (2007), the cross-correlation between the
    0p1s, 2p1s, and 1p2s depth traces measures the consistency of
    the three phases. A higher cross-correlation indicates that the
    three phases agree on the Moho depth for a given Vp/Vs ratio.

    Parameters
    ----------
    stack : np.ndarray
        Stacked amplitudes (nk, nh, 3) for the three phases
    nk : int
        Number of kappa values

    Returns
    -------
    cc_weight : np.ndarray
        Cross-correlation weight for each kappa (nk,)
    """
    cc_weight = np.ones(nk)
    for ik in range(nk):
        s0 = stack[ik, :, 0]  # 0p1s
        s1 = stack[ik, :, 1]  # 2p1s
        s2 = stack[ik, :, 2]  # 1p2s

        # Compute pairwise cross-correlations, handling zero-variance cases
        cc01 = 0.0
        cc02 = 0.0
        cc12 = 0.0

        std0, std1, std2 = np.std(s0), np.std(s1), np.std(s2)
        if std0 > 1e-10 and std1 > 1e-10:
            cc01 = np.corrcoef(s0, s1)[0, 1]
        if std0 > 1e-10 and std2 > 1e-10:
            cc02 = np.corrcoef(s0, s2)[0, 1]
        if std1 > 1e-10 and std2 > 1e-10:
            cc12 = np.corrcoef(s1, s2)[0, 1]

        # Average of pairwise correlations, clamped to >= 0
        cc_weight[ik] = max(0.0, (cc01 + cc02 + cc12) / 3.0)

    return cc_weight


def plot(stack, allstack, h, kappa, besth, bestk, cvalue, cmap=load_cyan_map(), title=None, path=None,
         plot_final_only=False):
    if plot_final_only:
        f, ax4 = plt.subplots(1, 1, figsize=(6, 5))
    else:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), sharex='col', sharey='row')
    xlim = (h[0], h[-1])
    ylim = (kappa[0], kappa[-1])
    if title is not None:
        f.suptitle(title, fontsize='large')
    if not plot_final_only:
        ax1.imshow(stack[:, :, 0], cmap=cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', origin='lower')
        ax1.set_ylabel('$V_P/V_S$')
        ax1.set_title('Ps')
        ax2.imshow(stack[:, :, 1], cmap=cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', origin='lower')
        ax2.set_title('PpPs')
        ax3.imshow(stack[:, :, 2], cmap=cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', origin='lower')
        ax3.set_title('PsPs+PpSs')
        ax3.set_xlabel('Moho depth (km)')
        ax3.set_ylabel('$V_P/V_S$')
    im = ax4.imshow(allstack, cmap=cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', origin='lower')
    ax4.plot(besth, bestk, color='red', marker='s', markerfacecolor='none')
    ax4.contour(allstack, [cvalue, 1], colors='k', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')
    ax4.plot(xlim, [bestk, bestk], color='red', linestyle='--', linewidth=0.6)
    ax4.plot([besth, besth], ylim, color='red', linestyle='--', linewidth=0.6)
    ax4.set_xlabel('Moho depth (km)')

    if plot_final_only:
        ax4.set_ylabel('$V_P/V_S$')
        plt.subplots_adjust(bottom=0.12, right=0.9, top=0.88)
    else:
        plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    _, yy, _, ww = ax4.get_position().bounds
    cax = plt.axes([0.93, yy, 0.016, ww])
    plt.colorbar(im, cax=cax)
    if path is None:
        plt.show()
    else:
        f.savefig(path, format='png', dpi=400, bbox_inches='tight')


def ci(allstack, h, kappa, ev_num):
    """
    Search best H and kappa from stacked matrix.
    Calculate error for H and kappa using Zhu & Kanamori (2000) contour method.

    :param allstack: stacked HK matrix
    :param h: 1-D array of H
    :param kappa: 1-D array of kappa
    :param ev_num: event number
    :return:
    """
    [i, j] = np.unravel_index(allstack.argmax(), allstack.shape)
    bestk = kappa[i]
    besth = h[j]

    cvalue = 1 - np.std(allstack.reshape(allstack.size)) / np.sqrt(ev_num)
    cs = plt.contour(h, kappa, allstack, [cvalue])
    if hasattr(cs, 'collections'):
        paths = cs.collections[0].get_paths()
        if len(paths) == 0:
            plt.close()
            raise ValueError('No contour path found for confidence level {:.4f}. '
                             'Try bootstrap method instead.'.format(cvalue))
        cs_path = paths[0].vertices
    else:
        if len(cs.allsegs) == 0 or len(cs.allsegs[0]) == 0:
            plt.close()
            raise ValueError('No contour path found for confidence level {:.4f}. '
                             'Try bootstrap method instead.'.format(cvalue))
        cs_path = cs.allsegs[0][0]
    maxhsig = (np.max(cs_path[:, 0]) - np.min(cs_path[:, 0])) / 2
    maxksig = (np.max(cs_path[:, 1]) - np.min(cs_path[:, 1])) / 2
    plt.close()
    return besth, bestk, cvalue, maxhsig, maxksig


def _bootstrap_one(args):
    """Single bootstrap iteration (module-level for multiprocessing)."""
    idx, seis, t0, dt, p, h, kappa, vp, weight, stack_method, nth_order = args
    seis_boot = seis[idx]
    p_boot = p[idx]
    _, _, normed, _ = hkstack(seis_boot, t0, dt, p_boot, h, kappa,
                               vp=vp, weight=weight,
                               stack_method=stack_method, nth_order=nth_order)
    i, j = np.unravel_index(normed.argmax(), normed.shape)
    return h[j], kappa[i]


def bootstrap_errors(seis, t0, dt, p, h, kappa, vp=6.3, weight=(0.7, 0.2, 0.1),
                     stack_method='linear', nth_order=4, n_bootstrap=100,
                     n_jobs=None):
    """Estimate H and kappa errors using parallel bootstrap resampling.

    Bootstrap method as used in Niu et al. (2007, JGR).
    Resamples RFs with replacement and recomputes H-k stack in parallel.

    Parameters
    ----------
    seis : np.ndarray
        RF data matrix (nrf x nt)
    t0, dt, p, h, kappa, vp, weight, stack_method, nth_order :
        Same as hkstack()
    n_bootstrap : int
        Number of bootstrap iterations (default 100)
    n_jobs : int or None
        Number of parallel workers (default: cpu_count - 1, min 1)

    Returns
    -------
    besth, bestk, maxhsig, maxksig
    """
    nrf = len(p)

    # Run full stack once to get best H, k
    _, _, normed_full, _ = hkstack(seis, t0, dt, p, h, kappa,
                                    vp=vp, weight=weight,
                                    stack_method=stack_method, nth_order=nth_order)
    i_full, j_full = np.unravel_index(normed_full.argmax(), normed_full.shape)
    bestk = kappa[i_full]
    besth = h[j_full]

    # Pre-generate bootstrap index sets
    rng = np.random.RandomState(42)
    idx_sets = [rng.choice(nrf, size=nrf, replace=True) for _ in range(n_bootstrap)]

    # Build task list
    tasks = [(idx_sets[ib], seis, t0, dt, p, h, kappa, vp, weight,
              stack_method, nth_order) for ib in range(n_bootstrap)]

    # Determine number of workers (cap at 8 — memory bandwidth saturates beyond)
    if n_jobs is None:
        n_jobs = min(8, max(1, (os.cpu_count() or 2) - 1))
    n_jobs = min(n_jobs, n_bootstrap)

    h_samples = np.zeros(n_bootstrap)
    k_samples = np.zeros(n_bootstrap)

    print(f'  Bootstrap: {n_bootstrap} iterations, {n_jobs} parallel workers ...')
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_bootstrap_one, t): ib for ib, t in enumerate(tasks)}
        done = 0
        for future in as_completed(futures):
            ib = futures[future]
            h_samples[ib], k_samples[ib] = future.result()
            done += 1
            if done % max(1, n_bootstrap // 10) == 0:
                print(f'    {done}/{n_bootstrap} done')

    maxhsig = np.std(h_samples)
    maxksig = np.std(k_samples)

    return besth, bestk, maxhsig, maxksig


def print_result(besth, bestk, maxhsig, maxksig, print_comment=True):
    header = 'H\tH_error\tk\tk_error\n'
    if print_comment:
        msg = '{}{:.1f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(header, besth, maxhsig, bestk, maxksig)
    else:
        msg = '{:.1f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(besth, maxhsig, bestk, maxksig)
    print(msg)


def hksta(hpara:HKPara, isplot=False, isdisplay=False):
    stadata = RFStation(hpara.rfpath, only_r=True)
    stack, _, allstack, _ = hkstack(stadata.data_prime, stadata.shift, stadata.sampling, srad2skm(stadata.rayp),
                                    hpara.hrange, hpara.krange, vp=hpara.vp, weight=hpara.weight,
                                    stack_method=hpara.stack_method, nth_order=hpara.nth_order)

    if hpara.use_bootstrap:
        besth, bestk, maxhsig, maxksig = bootstrap_errors(
            stadata.data_prime, stadata.shift, stadata.sampling, srad2skm(stadata.rayp),
            hpara.hrange, hpara.krange, vp=hpara.vp, weight=hpara.weight,
            stack_method=hpara.stack_method, nth_order=hpara.nth_order,
            n_bootstrap=hpara.n_bootstrap)
        cvalue = None
    else:
        besth, bestk, cvalue, maxhsig, maxksig = ci(allstack, hpara.hrange, hpara.krange, stadata.ev_num)
    with open(hpara.hklist, 'a') as f:
        f.write('{}\t{:.3f}\t{:.3f}\t{:.1f}\t{:.2f}\t{:.2f}\t{:.3f}\n'.format(stadata.staname, stadata.stla, stadata.stlo,
                                                                              besth, maxhsig, bestk, maxksig))
    title = '{}\nMoho depth = ${:.1f}\pm{:.2f}$ km\n$V_P/V_S$ = ${:.2f}\pm{:.3f}$'.format(stadata.staname, besth,
                                                                                     maxhsig, bestk, maxksig)
    if isdisplay:
        print_result(besth, bestk, maxhsig, maxksig, print_comment=True)

    if hpara.energy_grid != '':
        if np.ndim(allstack) != 2:
            raise ValueError('Normalized energy stack should be 2-D matrix')
        with open(hpara.energy_grid, 'w') as f:
            f.write('# H(km)\tk\tnormalized_energy\n')
            for i, k in enumerate(hpara.krange):
                for j, h in enumerate(hpara.hrange):
                    f.write('{:.2f}\t{:.3f}\t{:.6f}\n'.format(h, k, allstack[i, j]))

    if isplot:
        img_path = join(hpara.hkpath, stadata.staname+'_Hk.png')
        plot(stack, allstack, hpara.hrange, hpara.krange, besth, bestk,
             cvalue if cvalue is not None else 0.95,
             title=title, path=img_path, plot_final_only=hpara.plot_final_only)
    else:
        plot(stack, allstack, hpara.hrange, hpara.krange, besth, bestk,
             cvalue if cvalue is not None else 0.95,
             title=title, plot_final_only=hpara.plot_final_only)


def hk():
    parser = argparse.ArgumentParser(description="HK stacking for single station")
    parser.add_argument('cfg_file', type=str, help='Path to HK configure file')
    parser.add_argument('-v', help='Display results to standard output',
                        dest='isdisplay', action='store_true')
    parser.add_argument('--method', type=str, default=None,
                        choices=['linear', 'nth_root', 'cc', 'nth_root_cc'],
                        help='Stacking method: linear (default), nth_root (Niu et al. 2007 noise suppression), '
                             'cc (Niu et al. 2007 cross-correlation weighted), '
                             'nth_root_cc (combined nth-root + CC weighting). Overrides config file setting.')
    parser.add_argument('--nth-order', type=int, default=None,
                        help='Order of nth-root stacking (default 4, as in Niu et al. 2007). '
                             'Overrides config file setting.')
    arg = parser.parse_args()
    hpara = hkpara(arg.cfg_file)
    if arg.method is not None:
        hpara.stack_method = arg.method
    if arg.nth_order is not None:
        hpara.nth_order = arg.nth_order
    hksta(hpara, isplot=True, isdisplay=arg.isdisplay)


if __name__ == '__main__':
    pass
