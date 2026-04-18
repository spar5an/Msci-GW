"""
sky_time_conversion.py — convert model-frame RA to true-event RA.

The training pipeline generates every waveform at a single reference GPS time
(1126259462.4, GW150914) while varying (ra, dec, psi) across samples. A model
trained on that data therefore predicts RA *in the training frame* — i.e. it
outputs the RA that would produce the observed waveform if the event had
occurred at t_ref.

For a real event at t_event, the true celestial RA is rotated forward from the
model's prediction by the sidereal angle swept between t_ref and t_event.
`ra_from_reference_time(ra_hat, t_event)` applies that correction. Dec and psi
are unaffected by Earth rotation and need no conversion.

Antenna-pattern invariance guarantees that

    (ra_hat, t_ref)  and  (ra_from_reference_time(ra_hat, t_event), t_event)

describe the same physical source.

Not wired into the rest of the pipeline — import and call at your discretion.

Run directly to execute the tests:
    python sky_time_conversion.py
"""

from __future__ import annotations

import numpy as np
from pycbc.detector import Detector

# Reference training GPS time used by generate_dataset.py (GW150914 merger).
T_REF_DEFAULT = 1126259462.4

# Earth sidereal angular velocity. Sidereal day = 86164.0905 s.
OMEGA_EARTH = 2.0 * np.pi / 86164.0905  # rad/s


def ra_from_reference_time(
    ra_hat: np.ndarray | float,
    t_event: np.ndarray | float,
    t_ref: float = T_REF_DEFAULT,
) -> np.ndarray | float:
    """
    Convert a model's RA prediction (in the training frame at `t_ref`) into the
    true celestial RA at the event's actual GPS time `t_event`.

    Parameters
    ----------
    ra_hat : model-predicted right ascension(s) [rad], expressed in the
        training reference frame.
    t_event : actual GPS time(s) of the real event.
    t_ref : reference GPS time the model was trained at (default: GW150914).

    Returns
    -------
    ra_true : same shape as `ra_hat`, wrapped into [0, 2π).
    """
    ra_hat = np.asarray(ra_hat, dtype=np.float64)
    t_event = np.asarray(t_event, dtype=np.float64)
    return np.mod(ra_hat + OMEGA_EARTH * (t_event - t_ref), 2.0 * np.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def _antenna_FpFx(ra, dec, psi, t_gps, detectors=('H1', 'L1')):
    """Helper: stack (F+, Fx) for each detector into shape (N, 2*D)."""
    ra = np.atleast_1d(ra); dec = np.atleast_1d(dec)
    psi = np.atleast_1d(psi)
    N = ra.shape[0]
    t_gps = np.broadcast_to(np.atleast_1d(t_gps), (N,))
    out = np.empty((N, 2 * len(detectors)))
    for j, name in enumerate(detectors):
        det = Detector(name)
        for i in range(N):
            fp, fc = det.antenna_pattern(ra[i], dec[i], psi[i], float(t_gps[i]))
            out[i, 2 * j] = fp
            out[i, 2 * j + 1] = fc
    return out


def _test_identity_at_reference_time() -> None:
    """When t_event == t_ref, no conversion is needed: output must equal input."""
    rng = np.random.default_rng(0)
    ra_hat = rng.uniform(0, 2 * np.pi, 20)
    ra_out = ra_from_reference_time(ra_hat, T_REF_DEFAULT)
    np.testing.assert_allclose(ra_out, ra_hat, atol=1e-12)
    print("  [PASS] identity when t_event == t_ref")


def _test_conversion_recovers_true_sky() -> None:
    """Core property: a perfect model predicting ra_hat from a waveform
    generated at (ra_true, dec, psi, t_event) will output the training-frame
    RA, i.e. ra_hat = ra_true − ω·(t_event − t_ref). The conversion must
    recover ra_true exactly."""
    rng = np.random.default_rng(1)
    N = 50
    ra_true = rng.uniform(0, 2 * np.pi, N)
    dec = np.arcsin(rng.uniform(-1, 1, N))
    psi = rng.uniform(0, np.pi, N)
    dt = rng.uniform(-3 * 86400, 3 * 86400, N)
    t_event = T_REF_DEFAULT + dt

    # Simulate a perfect model: it outputs the ra that reproduces the
    # antenna pattern when paired with t_ref.
    ra_hat = np.mod(ra_true - OMEGA_EARTH * dt, 2 * np.pi)

    ra_recovered = ra_from_reference_time(ra_hat, t_event)

    # Compare in antenna-pattern space (robust to 2π wraparound).
    L_recovered = _antenna_FpFx(ra_recovered, dec, psi, t_event)
    L_truth     = _antenna_FpFx(ra_true,      dec, psi, t_event)
    diff = np.abs(L_recovered - L_truth).max()
    assert diff < 1e-3, f"recovered sky mismatch: max |ΔF| = {diff:.3e}"
    print(f"  [PASS] conversion recovers true sky: max |ΔF| = {diff:.3e}")


def _test_preserves_antenna_pattern_across_frames() -> None:
    """(ra_hat, t_ref) and (ra_from_reference_time(ra_hat, t_event), t_event)
    must describe the same physical source — identical antenna patterns."""
    rng = np.random.default_rng(2)
    N = 50
    ra_hat = rng.uniform(0, 2 * np.pi, N)
    dec = np.arcsin(rng.uniform(-1, 1, N))
    psi = rng.uniform(0, np.pi, N)
    t_event = T_REF_DEFAULT + rng.uniform(-3 * 86400, 3 * 86400, N)

    ra_event = ra_from_reference_time(ra_hat, t_event)

    L_train_frame = _antenna_FpFx(ra_hat,   dec, psi, T_REF_DEFAULT)
    L_event_frame = _antenna_FpFx(ra_event, dec, psi, t_event)
    diff = np.abs(L_train_frame - L_event_frame).max()
    mean = np.abs(L_train_frame - L_event_frame).mean()
    assert diff < 1e-3, f"frame mismatch: max |ΔF| = {diff:.3e}"
    print(f"  [PASS] antenna pattern invariant across frames: "
          f"max |ΔF| = {diff:.3e}, mean = {mean:.3e}")


def _test_conversion_is_needed() -> None:
    """Skipping the conversion (reading ra_hat as if it were the true RA)
    leaves large sky-position errors at t_event != t_ref."""
    rng = np.random.default_rng(3)
    N = 50
    ra_hat = rng.uniform(0, 2 * np.pi, N)
    dec = np.arcsin(rng.uniform(-1, 1, N))
    psi = rng.uniform(0, np.pi, N)
    t_event = T_REF_DEFAULT + rng.uniform(-3 * 86400, 3 * 86400, N)

    # What the correct answer is (after conversion):
    L_correct = _antenna_FpFx(ra_from_reference_time(ra_hat, t_event),
                              dec, psi, t_event)
    # What you'd get by not converting — pretending ra_hat is the true RA:
    L_naive = _antenna_FpFx(ra_hat, dec, psi, t_event)
    diff = np.abs(L_correct - L_naive).mean()
    assert diff > 0.1, f"expected large naive mismatch, got {diff:.3e}"
    print(f"  [PASS] without conversion, sky prediction is wrong: "
          f"mean |ΔF| = {diff:.3f}")


def _test_vectorization_and_shape() -> None:
    """Function must accept scalars and arrays; output shape must match input."""
    r = ra_from_reference_time(1.0, T_REF_DEFAULT + 3600)
    assert np.isscalar(r) or r.shape == (), f"scalar in -> scalar out, got shape {r.shape}"

    ra = np.linspace(0, 2 * np.pi, 7)
    out = ra_from_reference_time(ra, T_REF_DEFAULT + 3600)
    assert out.shape == (7,)
    assert np.all((out >= 0) & (out < 2 * np.pi)), "output must be in [0, 2π)"

    out2 = ra_from_reference_time(ra, np.full(7, T_REF_DEFAULT + 3600))
    np.testing.assert_allclose(out, out2)
    print("  [PASS] shape/vectorization correct")


def _test_full_day_is_near_identity() -> None:
    """One sidereal day later, Earth is back to the same orientation so the
    conversion must return the input (mod 2π)."""
    rng = np.random.default_rng(4)
    ra_hat = rng.uniform(0, 2 * np.pi, 10)
    t_event = T_REF_DEFAULT + 86164.0905
    ra_out = ra_from_reference_time(ra_hat, t_event)
    diff = np.mod(ra_hat - ra_out + np.pi, 2 * np.pi) - np.pi
    assert np.max(np.abs(diff)) < 1e-10, f"sidereal-day round trip failed: {diff}"
    print("  [PASS] one sidereal day offset returns RA unchanged")


if __name__ == '__main__':
    print("Testing ra_from_reference_time …")
    _test_identity_at_reference_time()
    _test_vectorization_and_shape()
    _test_full_day_is_near_identity()
    _test_conversion_is_needed()
    _test_preserves_antenna_pattern_across_frames()
    _test_conversion_recovers_true_sky()
    print("All tests passed.")
