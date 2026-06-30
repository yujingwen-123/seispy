"""
Test cases for the new H-k stacking methods:
- nth_root: Nth-root stacking for noise suppression (Niu et al., 2007)
- cc: Cross-correlation weighted stacking (Niu et al., 2007)
- nth_root_cc: Combined nth-root + cross-correlation weighting

Tests use synthetic RFs with known Moho depth and Vp/Vs ratio.
"""
import numpy as np
from seispy.hk import hkstack, _compute_cc_weight


def make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75, vp=6.3,
                       dt=0.1, t0=5.0, noise_level=0.05):
    """Generate synthetic receiver functions with known Moho parameters.

    Uses Ricker wavelet-like pulses at the predicted arrival times for
    0p1s, 2p1s, and 1p2s phases.
    """
    nt = 600  # 60 seconds at 0.1s sampling
    t = np.arange(nt) * dt - t0
    vs = vp / k_true

    # Ray parameters between 0.04 and 0.08 s/km
    p = np.linspace(0.04, 0.08, n_rf)

    seis = np.zeros((n_rf, nt))
    for i in range(n_rf):
        eta_p = np.sqrt(1/vp**2 - p[i]**2)
        eta_s = np.sqrt(1/vs**2 - p[i]**2)

        # Predicted arrival times
        t_ps = h_true * (eta_s - eta_p)
        t_ppps = h_true * (eta_s + eta_p)
        t_psps = h_true * (2 * eta_s)

        # Ricker wavelet
        for tau, amp in [(t_ps, 1.0), (t_ppps, 0.7), (t_psps, -0.5)]:
            t_rel = t - tau
            # Ricker wavelet
            freq = 1.5
            wavelet = (1 - 2 * (np.pi * freq * t_rel)**2) * np.exp(-(np.pi * freq * t_rel)**2)
            seis[i] += amp * wavelet

        # Add noise
        seis[i] += noise_level * np.random.randn(nt)

    return seis, p, t0, dt


def test_linear_consistency():
    """Test that linear stacking finds the correct Moho parameters."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    _, _, normed, _ = hkstack(seis, t0, dt, p, h, kappa,
                               vp=6.3, stack_method='linear')

    i, j = np.unravel_index(normed.argmax(), normed.shape)
    best_k = kappa[i]
    best_h = h[j]

    print(f"Linear: best H={best_h:.1f}, best k={best_k:.3f}")
    assert abs(best_h - 40.0) < 3.0, f"H mismatch: {best_h} vs 40.0"
    assert abs(best_k - 1.75) < 0.1, f"k mismatch: {best_k} vs 1.75"
    print("  PASSED")


def test_nth_root_consistency():
    """Test that nth-root stacking finds the correct Moho parameters."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    _, _, normed, _ = hkstack(seis, t0, dt, p, h, kappa,
                               vp=6.3, stack_method='nth_root', nth_order=4)

    i, j = np.unravel_index(normed.argmax(), normed.shape)
    best_k = kappa[i]
    best_h = h[j]

    print(f"Nth-root (N=4): best H={best_h:.1f}, best k={best_k:.3f}")
    assert abs(best_h - 40.0) < 3.0, f"H mismatch: {best_h} vs 40.0"
    assert abs(best_k - 1.75) < 0.1, f"k mismatch: {best_k} vs 1.75"
    print("  PASSED")


def test_nth_root_noise_suppression():
    """Test that nth-root stacking suppresses high-amplitude noise better than linear stacking.

    Add an outlier RF with very high noise and check that nth-root handles it better.
    """
    np.random.seed(123)
    n_rf = 30
    h_true, k_true = 40.0, 1.75
    vp = 6.3
    dt = 0.1
    t0 = 5.0
    nt = 600
    t = np.arange(nt) * dt - t0
    vs = vp / k_true

    p = np.linspace(0.04, 0.08, n_rf)
    seis = np.zeros((n_rf, nt))
    for i in range(n_rf):
        eta_p = np.sqrt(1/vp**2 - p[i]**2)
        eta_s = np.sqrt(1/vs**2 - p[i]**2)

        t_ps = h_true * (eta_s - eta_p)
        t_ppps = h_true * (eta_s + eta_p)
        t_psps = h_true * (2 * eta_s)

        for tau, amp in [(t_ps, 1.0), (t_ppps, 0.7), (t_psps, -0.5)]:
            t_rel = t - tau
            freq = 1.5
            wavelet = (1 - 2 * (np.pi * freq * t_rel)**2) * np.exp(-(np.pi * freq * t_rel)**2)
            seis[i] += amp * wavelet

        # Normal noise for most traces
        if i < n_rf - 2:
            seis[i] += 0.05 * np.random.randn(nt)
        else:
            # Add very high noise to last 2 traces
            seis[i] += 2.0 * np.random.randn(nt)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    # Linear stacking (affected by outliers)
    _, _, normed_lin, _ = hkstack(seis, t0, dt, p, h, kappa,
                                    vp=vp, stack_method='linear')
    i_lin, j_lin = np.unravel_index(normed_lin.argmax(), normed_lin.shape)

    # Nth-root stacking (robust to outliers)
    _, _, normed_nth, _ = hkstack(seis, t0, dt, p, h, kappa,
                                    vp=vp, stack_method='nth_root', nth_order=4)
    i_nth, j_nth = np.unravel_index(normed_nth.argmax(), normed_nth.shape)

    err_lin = abs(h[j_lin] - h_true)
    err_nth = abs(h[j_nth] - h_true)

    print(f"With outliers: Linear H error={err_lin:.1f} km, Nth-root H error={err_nth:.1f} km")
    # Nth-root should be at least as accurate as linear
    # (with strong outliers, nth-root is typically more accurate)
    assert err_nth <= err_lin + 0.5, \
        f"Nth-root should be similar or better with outliers. Linear: {err_lin}, Nth-root: {err_nth}"
    print("  PASSED")


def test_cc_weight():
    """Test that _compute_cc_weight works correctly."""
    np.random.seed(99)
    nk, nh = 5, 200

    # Create stacks where modes are correlated (should give high CC weight).
    # In real H-k stacking, the three phase stacks all peak positively at
    # the Moho depth when Vp/Vs is correct (the negative sign of 1p2s is
    # already accounted for in tstack[:,:,2] = -am_cor * seis[i, t3]).
    # So the three mode stacks should be positively correlated.
    base = np.sin(np.linspace(0, 4*np.pi, nh))
    stack_correlated = np.zeros((nk, nh, 3))
    for ik in range(nk):
        stack_correlated[ik, :, 0] = base + 0.3 * np.random.randn(nh)
        stack_correlated[ik, :, 1] = 0.7 * base + 0.3 * np.random.randn(nh)
        stack_correlated[ik, :, 2] = 0.5 * base + 0.3 * np.random.randn(nh)

    cc_corr = _compute_cc_weight(stack_correlated, nk)
    print(f"Correlated traces CC weights: {cc_corr}")
    assert np.all(cc_corr > 0.3), f"Correlated traces should have high CC weight, got {cc_corr}"

    # Create stacks where modes are uncorrelated
    stack_uncorrelated = np.zeros((nk, nh, 3))
    for ik in range(nk):
        stack_uncorrelated[ik, :, 0] = np.sin(np.linspace(0, 4*np.pi, nh)) + 0.5 * np.random.randn(nh)
        stack_uncorrelated[ik, :, 1] = np.cos(np.linspace(0, 3*np.pi, nh)) + 0.5 * np.random.randn(nh)
        stack_uncorrelated[ik, :, 2] = np.sin(np.linspace(0, 7*np.pi, nh)) + 0.5 * np.random.randn(nh)

    cc_uncorr = _compute_cc_weight(stack_uncorrelated, nk)
    print(f"Uncorrelated traces CC weights: {cc_uncorr}")
    # Uncorrelated should have lower CC than correlated
    assert np.mean(cc_corr) > np.mean(cc_uncorr), \
        f"Correlated ({np.mean(cc_corr):.3f}) should have higher CC weight than uncorrelated ({np.mean(cc_uncorr):.3f})"
    print("  PASSED")


def test_cc_stacking():
    """Test that CC-weighted stacking finds the correct Moho parameters."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    _, _, normed, _ = hkstack(seis, t0, dt, p, h, kappa,
                               vp=6.3, stack_method='cc')

    i, j = np.unravel_index(normed.argmax(), normed.shape)
    best_k = kappa[i]
    best_h = h[j]

    print(f"CC-weighted: best H={best_h:.1f}, best k={best_k:.3f}")
    assert abs(best_h - 40.0) < 3.0, f"H mismatch: {best_h} vs 40.0"
    assert abs(best_k - 1.75) < 0.1, f"k mismatch: {best_k} vs 1.75"
    print("  PASSED")


def test_nth_root_cc_combined():
    """Test that combined nth-root + CC weighting works."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    _, _, normed, _ = hkstack(seis, t0, dt, p, h, kappa,
                               vp=6.3, stack_method='nth_root_cc', nth_order=4)

    i, j = np.unravel_index(normed.argmax(), normed.shape)
    best_k = kappa[i]
    best_h = h[j]

    print(f"Nth-root+CC: best H={best_h:.1f}, best k={best_k:.3f}")
    assert abs(best_h - 40.0) < 3.0, f"H mismatch: {best_h} vs 40.0"
    assert abs(best_k - 1.75) < 0.1, f"k mismatch: {best_k} vs 1.75"
    print("  PASSED")


def test_different_nth_orders():
    """Test nth-root stacking with different orders."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    for n in [2, 4, 6]:
        _, _, normed, _ = hkstack(seis, t0, dt, p, h, kappa,
                                   vp=6.3, stack_method='nth_root', nth_order=n)
        i, j = np.unravel_index(normed.argmax(), normed.shape)
        best_k = kappa[i]
        best_h = h[j]
        print(f"Nth-root (N={n}): best H={best_h:.1f}, best k={best_k:.3f}")
        assert abs(best_h - 40.0) < 3.0, f"N={n}: H mismatch: {best_h} vs 40.0"
    print("  PASSED")


def test_backward_compatibility():
    """Test that default parameters maintain backward compatibility."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=30, h_true=40.0, k_true=1.75)

    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)

    # Default call (no stack_method specified)
    _, _, normed_default, _ = hkstack(seis, t0, dt, p, h, kappa, vp=6.3)

    # Explicit linear
    _, _, normed_linear, _ = hkstack(seis, t0, dt, p, h, kappa,
                                       vp=6.3, stack_method='linear')

    # Results should be identical
    assert np.allclose(normed_default, normed_linear), \
        "Default and explicit 'linear' should give identical results"
    print("Backward compatibility: PASSED")


def test_return_shapes():
    """Test that all methods return correctly shaped arrays."""
    np.random.seed(42)
    seis, p, t0, dt = make_synthetic_rf(n_rf=20, h_true=40.0, k_true=1.75)

    nrf = len(p)
    h = np.arange(30, 50, 0.1)
    kappa = np.arange(1.6, 1.9, 0.01)
    nh, nk = len(h), len(kappa)

    for method in ['linear', 'nth_root', 'cc', 'nth_root_cc']:
        stack, stackvar, normed, allstackvar = hkstack(
            seis, t0, dt, p, h, kappa, vp=6.3, stack_method=method)

        assert stack.shape == (nk, nh, 3), \
            f"{method}: stack shape {stack.shape} != ({nk}, {nh}, 3)"
        assert stackvar.shape == (nk, nh, 3), \
            f"{method}: stackvar shape {stackvar.shape} != ({nk}, {nh}, 3)"
        assert normed.shape == (nk, nh), \
            f"{method}: normed shape {normed.shape} != ({nk}, {nh})"
        assert allstackvar.shape == (nk, nh), \
            f"{method}: allstackvar shape {allstackvar.shape} != ({nk}, {nh})"
        print(f"{method}: shapes OK")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing H-k stacking methods (Niu et al., 2007)")
    print("=" * 60)

    print("\n1. Linear stacking consistency:")
    test_linear_consistency()

    print("\n2. Nth-root stacking consistency:")
    test_nth_root_consistency()

    print("\n3. Nth-root noise suppression:")
    test_nth_root_noise_suppression()

    print("\n4. CC weight computation:")
    test_cc_weight()

    print("\n5. CC-weighted stacking:")
    test_cc_stacking()

    print("\n6. Combined nth-root + CC:")
    test_nth_root_cc_combined()

    print("\n7. Different nth-root orders:")
    test_different_nth_orders()

    print("\n8. Backward compatibility:")
    test_backward_compatibility()

    print("\n9. Return shapes:")
    test_return_shapes()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
