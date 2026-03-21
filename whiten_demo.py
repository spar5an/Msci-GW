#!/usr/bin/env python3
"""
whiten_demo.py — Generate noisy GR gravitational wave data and whiten it.

Pipeline:
  1. Generate a clean GW150914-like signal (IMRPhenomD, H1, FD->IRFFT)
     using the same pipeline as gw_datagen._generate_single_waveform
  2. Add O4-era coloured Gaussian noise (real O4a PSD via o4_psd.py)
  3. Whiten the noisy strain via gw_datagen.whiten_waveform
  4. Plot time-domain comparison and PSD verification
  5. Print whitening effectiveness metrics

Outputs (saved alongside this script):
  whiten_demo_time.png   — time domain: noisy / clean / whitened
  whiten_demo_psd.png    — PSD verification: before/after whitening
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch as sp_welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gw_datagen import whiten_waveform, _apply_end_taper, _HIGHPASS_FC
from pycbc.waveform import get_fd_waveform
from pycbc.types import TimeSeries
from pycbc.filter import highpass_fir
from pycbc.detector import Detector
from pycbc.noise import noise_from_psd
from o4_psd import load_random_o4_psd

# ---------------------------------------------------------------------------
# Parameters (match gw_datagen / plot_waveform defaults)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 4096
DELTA_T     = 1.0 / SAMPLE_RATE
F_LOWER     = 20.0
F_FINAL     = 2048.0
DELTA_F_FD  = 1.0 / 256       # FD resolution used in gw_datagen
N_RINGDOWN  = 500              # merger + early ringdown samples kept at head
TARGET_LEN  = 16384            # 4 s at 4096 Hz
GPS_TIME    = 1126259462.4     # GW150914 epoch (same default as gw_datagen)
DETECTOR    = 'H1'

# Source parameters (GW150914-like)
MASS1    = 36.0    # M_sun
MASS2    = 29.0    # M_sun
DISTANCE = 410.0   # Mpc

NOISE_SEED  = 42
SHOW_WINDOW = (-1.5, 0.15)    # seconds around merger for time-domain panels

# Merger sits at this sample index after the FD->IRFFT assembly
_MERGER_IDX = TARGET_LEN - N_RINGDOWN
TIMES = (np.arange(TARGET_LEN) - _MERGER_IDX) * DELTA_T   # relative to merger

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Step 1: Generate clean GW signal (FD -> IRFFT pipeline)
# Mirrors _generate_single_waveform / plot_waveform.py exactly.
# ---------------------------------------------------------------------------
print(f"[1/4] Generating clean GW signal  ({MASS1:.0f}+{MASS2:.0f} M☉, "
      f"d={DISTANCE} Mpc, {DETECTOR}, IMRPhenomD) ...")

hp_fd, hc_fd = get_fd_waveform(
    approximant='IMRPhenomD',
    mass1=MASS1,
    mass2=MASS2,
    spin1z=0.0,
    spin2z=0.0,
    inclination=0.0,
    coa_phase=0.0,
    distance=DISTANCE,
    delta_f=DELTA_F_FD,
    f_lower=F_LOWER,
    f_final=F_FINAL,
)

hp_raw = np.fft.irfft(hp_fd.numpy())
hc_raw = np.fft.irfft(hc_fd.numpy())
N_fd   = len(hp_raw)
hp_raw *= DELTA_F_FD * N_fd
hc_raw *= DELTA_F_FD * N_fd

# Assemble to TARGET_LEN: keep last n_pre + first N_RINGDOWN samples
if TARGET_LEN <= N_fd:
    n_pre  = TARGET_LEN - N_RINGDOWN
    hp_arr = np.concatenate([hp_raw[N_fd - n_pre:], hp_raw[:N_RINGDOWN]])
    hc_arr = np.concatenate([hc_raw[N_fd - n_pre:], hc_raw[:N_RINGDOWN]])
else:
    hp_arr = np.concatenate([np.zeros(TARGET_LEN - N_fd), hp_raw])
    hc_arr = np.concatenate([np.zeros(TARGET_LEN - N_fd), hc_raw])

hp_arr = _apply_end_taper(hp_arr)
hc_arr = _apply_end_taper(hc_arr)

hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=DELTA_T)
hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=DELTA_T)
hp_ts.start_time += GPS_TIME
hc_ts.start_time += GPS_TIME
hp_ts = highpass_fir(hp_ts, _HIGHPASS_FC, 128)
hc_ts = highpass_fir(hc_ts, _HIGHPASS_FC, 128)

det    = Detector(DETECTOR)
sig_ts = det.project_wave(hp_ts, hc_ts, ra=0.0, dec=np.pi / 2,
                           polarization=0.0, method='lal')

sig = sig_ts.numpy()
if len(sig) >= TARGET_LEN:
    sig = sig[len(sig) - TARGET_LEN:]
else:
    sig = np.concatenate([np.zeros(TARGET_LEN - len(sig), dtype=sig.dtype), sig])

print(f"      Clean signal peak strain: {np.max(np.abs(sig)):.3e}")

# ---------------------------------------------------------------------------
# Step 2: Generate O4-era coloured Gaussian noise
# ---------------------------------------------------------------------------
print(f"[2/4] Generating coloured noise  (O4a real PSD via o4_psd, seed={NOISE_SEED}) ...")

plen    = TARGET_LEN // 2 + 1
delta_f = 1.0 / (TARGET_LEN * DELTA_T)
psd_ref = load_random_o4_psd(plen, delta_f, F_LOWER, DETECTOR, SAMPLE_RATE,
                              rng=np.random.default_rng(NOISE_SEED))
noise_ts = noise_from_psd(TARGET_LEN, DELTA_T, psd_ref, seed=NOISE_SEED)
noise = noise_ts.numpy()[:TARGET_LEN]

print(f"      Noise RMS: {np.std(noise):.3e}")

# ---------------------------------------------------------------------------
# Step 3: Noisy signal = signal + noise, then highpass to remove sub-band
# o4_psd sets PSD = 1e40 below f_lower (sentinel for whitening), which means
# noise_from_psd creates enormous sub-band noise. Highpassing before analysis
# mirrors how real GW data is treated and gives a physically realistic noise floor.
# ---------------------------------------------------------------------------
noisy_raw = sig + noise

noisy_ts = TimeSeries(noisy_raw.astype(np.float64), delta_t=DELTA_T)
noisy_ts  = highpass_fir(noisy_ts, _HIGHPASS_FC, 128)
noisy     = np.array(noisy_ts)

snr_raw = np.max(np.abs(sig)) / np.std(noisy)
print(f"      Noise RMS (post-highpass): {np.std(noisy):.3e}")
print(f"      Pre-whitening signal peak / noise σ: {snr_raw:.3f}  (signal buried in noise)")

# ---------------------------------------------------------------------------
# Step 4: Whiten the noisy signal using gw_datagen.whiten_waveform
# Internally: Welch PSD estimate -> divide by sqrt(PSD) -> 35-300 Hz bandpass
# ---------------------------------------------------------------------------
print("[3/4] Whitening ...")

whitened, psd_arr, freq_arr = whiten_waveform(
    noisy,
    delta_t=DELTA_T,
    f_lower=F_LOWER,
    apply_bandpass=True,
    apply_tukey=True,
    tukey_alpha=0.1,
    tukey_side='left',
)

# Build a reference "whitened clean signal" by dividing the clean signal by
# sqrt(noise PSD) in the frequency domain — same PSD used for the noisy data.
sig_fft    = np.fft.rfft(sig)
freqs_sig  = np.fft.rfftfreq(len(sig), d=DELTA_T)
psd_interp = np.interp(freqs_sig, freq_arr, psd_arr, left=0.0, right=0.0)

with np.errstate(divide='ignore', invalid='ignore'):
    sig_white_fd = np.where(psd_interp > 0,
                            sig_fft / np.sqrt(np.maximum(psd_interp, 1e-100)),
                            0.0 + 0.0j)

# Apply same 35-300 Hz bandpass as whiten_waveform
sig_white_fd[(freqs_sig < 35) | (freqs_sig > 300)] = 0.0
whitened_clean = np.fft.irfft(sig_white_fd, n=len(sig))

# ---------------------------------------------------------------------------
# Whitening effectiveness metrics
# ---------------------------------------------------------------------------
print("[4/4] Computing whitening effectiveness metrics ...")

# Welch PSD of noisy and whitened signals
nperseg = SAMPLE_RATE  # 1-second segments -> 1 Hz resolution
f_n, pxx_n = sp_welch(noisy,    fs=SAMPLE_RATE, nperseg=nperseg)
f_w, pxx_w = sp_welch(whitened, fs=SAMPLE_RATE, nperseg=nperseg)

band_n = (f_n >= 35) & (f_n <= 300)
band_w = (f_w >= 35) & (f_w <= 300)

cv_pre  = np.std(pxx_n[band_n]) / np.mean(pxx_n[band_n])
cv_post = np.std(pxx_w[band_w]) / np.mean(pxx_w[band_w])

# SNR proxy: first 2 s are noise-only (inspiral starts ~1.5 s before merger)
n_quiet = 2 * SAMPLE_RATE   # samples in the quiet (noise-only) window
noise_rms_pre  = np.std(noisy[:n_quiet])
noise_rms_post = np.std(whitened[:n_quiet])
sig_peak_pre   = np.max(np.abs(noisy[n_quiet:]))
sig_peak_post  = np.max(np.abs(whitened[n_quiet:]))
snr_pre        = sig_peak_pre  / noise_rms_pre
snr_post       = sig_peak_post / noise_rms_post

# ASD flatness in-band: compare max/min ratio (ideal = 1)
asd_w_band = np.sqrt(pxx_w[band_w])
flatness_ratio = np.max(asd_w_band) / np.min(asd_w_band)

print()
print("=" * 60)
print("  Whitening Effectiveness Check")
print("=" * 60)
print(f"  Pre-whitening  PSD CV (35–300 Hz): {cv_pre:.3f}"
      "  (higher = more coloured)")
print(f"  Post-whitening PSD CV (35–300 Hz): {cv_post:.3f}"
      "  (lower = flatter)")
print(f"  Flatness improvement factor       : {cv_pre / cv_post:.1f}x")
print(f"  In-band ASD max/min ratio         : {flatness_ratio:.2f}"
      "  (1.0 = perfectly flat)")
print()
print(f"  Peak-to-noise (SNR proxy) pre-whitening : {snr_pre:.2f} σ")
print(f"  Peak-to-noise (SNR proxy) post-whitening: {snr_post:.2f} σ"
      "  (target > ~5 for GW150914-like)")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# Figure 1: Time Domain Comparison
# ---------------------------------------------------------------------------
mask   = (TIMES >= SHOW_WINDOW[0]) & (TIMES <= SHOW_WINDOW[1])
t_plot = TIMES[mask]

fig1, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig1.suptitle(
    rf"GW150914-like Signal  ({MASS1:.0f}+{MASS2:.0f} $M_\odot$, "
    rf"{DISTANCE} Mpc, {DETECTOR}, IMRPhenomD)"
    "\nReal O4a noise (highpassed >35 Hz) + PSD whitening (35–300 Hz bandpass)",
    fontsize=12,
)

# Panel 1: raw noisy strain
axes[0].plot(t_plot, noisy[mask] * 1e21,
             color='#555555', linewidth=0.5, alpha=0.9)
axes[0].axvline(0, color='firebrick', linewidth=0.9, linestyle='--',
                alpha=0.7, label='merger')
axes[0].set_ylabel(r'$h \times 10^{21}$', fontsize=10)
axes[0].set_title('Raw noisy detector strain  —  signal buried in coloured noise',
                  fontsize=10)
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, alpha=0.3, linewidth=0.4)

# Panel 2: clean signal reference
axes[1].plot(t_plot, sig[mask] * 1e21,
             color='#1f6fbf', linewidth=1.1)
axes[1].axvline(0, color='firebrick', linewidth=0.9, linestyle='--', alpha=0.7)
axes[1].set_ylabel(r'$h \times 10^{21}$', fontsize=10)
axes[1].set_title('Clean GW signal (no noise)  —  reference', fontsize=10)
axes[1].grid(True, alpha=0.3, linewidth=0.4)

# Panel 3: whitened noisy + reference
axes[2].plot(t_plot, whitened[mask],
             color='#2ca02c', linewidth=0.7, alpha=0.9, label='whitened noisy')
axes[2].plot(t_plot, whitened_clean[mask],
             color='#e05c00', linewidth=1.3, linestyle='--',
             alpha=0.85, label='expected whitened signal')
axes[2].axvline(0, color='firebrick', linewidth=0.9, linestyle='--', alpha=0.7)
axes[2].axhline(0, color='grey', linewidth=0.4, alpha=0.4)
axes[2].set_ylabel('Whitened strain\n(normalised units)', fontsize=10)
axes[2].set_title(
    f'Whitened noisy strain  —  GW chirp visible  '
    f'(SNR proxy: {snr_post:.1f} σ)',
    fontsize=10,
)
axes[2].legend(loc='upper left', fontsize=9)
axes[2].grid(True, alpha=0.3, linewidth=0.4)
axes[2].set_xlabel('Time from merger (s)', fontsize=10)
axes[2].set_xlim(*SHOW_WINDOW)

for ax in axes:
    ax.tick_params(labelsize=9)

plt.tight_layout()
out1 = os.path.join(SAVE_DIR, 'whiten_demo_time.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved {out1}")

# ---------------------------------------------------------------------------
# Figure 2: Whitening Verification (PSD / ASD panels)
# ---------------------------------------------------------------------------
whiten_filter = np.where(psd_arr > 0, 1.0 / np.sqrt(psd_arr), 0.0)

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
fig2.suptitle(
    "Whitening Effectiveness Verification\n"
    rf"PSD coeff. of variation: {cv_pre:.3f} $\to$ {cv_post:.3f}  "
    rf"({cv_pre/cv_post:.1f}$\times$ improvement)  |  "
    rf"in-band ASD max/min: {flatness_ratio:.2f}  |  "
    rf"SNR proxy: {snr_pre:.1f}$\sigma$ $\to$ {snr_post:.1f}$\sigma$",
    fontsize=11,
)

# Top-left: ASD of noisy signal
ax = axes2[0, 0]
mask_pos_n = f_n > 10
ax.semilogy(f_n[mask_pos_n], np.sqrt(pxx_n[mask_pos_n]),
            color='#555555', linewidth=1.0)
ax.axvspan(35, 300, alpha=0.08, color='green', label='35–300 Hz band')
ax.set_xlim(10, 1000)
ax.set_xlabel('Frequency (Hz)', fontsize=9)
ax.set_ylabel(r'ASD  (strain / $\sqrt{\mathrm{Hz}}$)', fontsize=9)
ax.set_title('ASD — noisy signal  (coloured noise spectrum)', fontsize=10)
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, which='both')

# Top-right: ASD of whitened signal
ax = axes2[0, 1]
mask_band_w_plot = (f_w >= 25) & (f_w <= 500)
ax.plot(f_w[mask_band_w_plot], np.sqrt(pxx_w[mask_band_w_plot]),
        color='#2ca02c', linewidth=1.0)
mean_asd_w = np.mean(np.sqrt(pxx_w[band_w]))
ax.axhline(mean_asd_w, color='firebrick', linewidth=0.9, linestyle='--',
           alpha=0.8, label=f'band mean = {mean_asd_w:.3f}')
ax.axvspan(35, 300, alpha=0.08, color='green', label='35–300 Hz band')
ax.set_xlim(25, 500)
ax.set_xlabel('Frequency (Hz)', fontsize=9)
ax.set_ylabel('ASD  (whitened units)', fontsize=9)
ax.set_title('ASD — whitened signal  (flat = effective whitening)', fontsize=10)
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# Bottom-left: ASD overlay (normalised)
ax = axes2[1, 0]
mask_ov = (f_n >= 20) & (f_n <= 500)
asd_n_ov = np.sqrt(pxx_n[mask_ov])
asd_w_ov = np.sqrt(pxx_w[(f_w >= 20) & (f_w <= 500)])
norm_n   = np.max(asd_n_ov)
norm_w   = np.max(asd_w_ov)
ax.semilogy(f_n[mask_ov], asd_n_ov / norm_n,
            color='#555555', linewidth=1.0, label='Pre-whitening (norm.)')
ax.plot(f_w[(f_w >= 20) & (f_w <= 500)], asd_w_ov / norm_w,
        color='#2ca02c', linewidth=1.0, label='Post-whitening (norm.)')
ax.axvspan(35, 300, alpha=0.08, color='green')
ax.set_xlim(20, 500)
ax.set_xlabel('Frequency (Hz)', fontsize=9)
ax.set_ylabel('Normalised ASD', fontsize=9)
ax.set_title('ASD overlay  —  before vs after whitening', fontsize=10)
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, which='both')

# Bottom-right: whitening filter shape 1/sqrt(PSD)
ax = axes2[1, 1]
fmask = (freq_arr >= 20) & (freq_arr <= 500) & (psd_arr > 0)
ax.semilogy(freq_arr[fmask], whiten_filter[fmask],
            color='#9467bd', linewidth=1.0)
ax.axvspan(35, 300, alpha=0.08, color='green', label='35–300 Hz band')
ax.set_xlabel('Frequency (Hz)', fontsize=9)
ax.set_ylabel(r'$1/\sqrt{S_n(f)}$  (whitening gain)', fontsize=9)
ax.set_title('Whitening filter shape  (estimated from O4a noise)', fontsize=10)
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, which='both')

for ax in axes2.flat:
    ax.tick_params(labelsize=8)

plt.tight_layout()
out2 = os.path.join(SAVE_DIR, 'whiten_demo_psd.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved {out2}")

try:
    plt.show()
except Exception:
    pass
