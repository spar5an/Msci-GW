# train_model_cpu.py
# CPU-only training for the DINGO model.
# Loads dataset.pt produced by generate_dataset.py (no GPU code, no AMP).
#
# Input:  dataset.pt  (written by Data Generation/generate_dataset.py)
# Output: <model_name>_cpu.pt

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")

DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

pi = np.pi
LOG_2PI = math.log(2.0 * math.pi)


# ==============================================================================
# NEURAL NETWORK CLASSES
# ==============================================================================

class AffineCouplingLayer(nn.Module):
    """Affine coupling with LayerNorm (batch-size-independent)."""
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1

        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, context, reverse=False):
        masked_x = x * self.mask
        scale_input = torch.cat([masked_x, context], dim=1)
        translation_input = torch.cat([masked_x, context], dim=1)

        s = self.scale_net(scale_input)
        t = self.translation_net(translation_input)

        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            y = x * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            y = (x - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        return y, log_det


def _searchsorted(bin_locations, inputs):
    """Return i s.t. bin_locations[..., i] <= inputs < bin_locations[..., i+1]."""
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def _rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights,
                               unnormalized_derivatives, inverse=False,
                               tail_bound=3.0, min_bin_width=1e-3,
                               min_bin_height=1e-3, min_derivative=1e-3):
    """Monotonic rational-quadratic spline with linear tails (Durkan 2019).

    Inputs have shape (..., D). The three unnormalized_* tensors have shape
    (..., D, num_bins) (widths, heights) and (..., D, num_bins - 1)
    (interior derivatives). Outside [-tail_bound, tail_bound] the map is the
    identity (log_det contribution = 0), so the flow still covers the real
    line. Returns (outputs, log_det) with log_det shape (..., D).
    """
    num_bins = unnormalized_widths.shape[-1]
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside = ~inside

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Tail: identity. Matches Durkan's "linear" tail default with derivative 1.
    outputs[outside] = inputs[outside]
    logabsdet[outside] = 0.0

    if inside.any():
        inp_in = inputs[inside]
        uw = unnormalized_widths[inside]
        uh = unnormalized_heights[inside]
        ud = unnormalized_derivatives[inside]

        # Normalise widths / heights to sum to 2*tail_bound, then soften floor.
        widths  = torch.softmax(uw, dim=-1)
        widths  = min_bin_width + (1 - min_bin_width * num_bins) * widths
        heights = torch.softmax(uh, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

        cumwidths  = torch.cumsum(widths,  dim=-1)
        cumheights = torch.cumsum(heights, dim=-1)
        cumwidths  = F.pad(cumwidths,  pad=(1, 0), mode='constant', value=0.0)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumwidths  = cumwidths  * (2 * tail_bound) - tail_bound
        cumheights = cumheights * (2 * tail_bound) - tail_bound
        cumwidths[...,  0] = -tail_bound; cumwidths[...,  -1] = tail_bound
        cumheights[..., 0] = -tail_bound; cumheights[..., -1] = tail_bound
        widths  = cumwidths[...,  1:] - cumwidths[...,  :-1]
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        derivatives = min_derivative + F.softplus(ud)
        # Boundary derivatives = 1 → linear tail matches smoothly.
        derivatives = F.pad(derivatives, pad=(1, 1), value=1.0)

        locations = cumheights if inverse else cumwidths
        bin_idx = _searchsorted(locations, inp_in).clamp(0, num_bins - 1)[..., None]
        input_cumwidths  = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        input_heights    = heights.gather(-1, bin_idx)[..., 0]
        delta = input_heights / input_bin_widths
        input_deriv      = derivatives.gather(-1, bin_idx)[..., 0]
        input_deriv_plus = derivatives.gather(-1, bin_idx + 1)[..., 0]

        if inverse:
            a = (inp_in - input_cumheights) * (input_deriv + input_deriv_plus - 2 * delta) \
                + input_heights * (delta - input_deriv)
            b = input_heights * input_deriv \
                - (inp_in - input_cumheights) * (input_deriv + input_deriv_plus - 2 * delta)
            c = -delta * (inp_in - input_cumheights)
            disc = b.pow(2) - 4 * a * c
            disc = disc.clamp(min=0.0)
            xi = (2 * c) / (-b - torch.sqrt(disc))
            out = xi * input_bin_widths + input_cumwidths
            theta_one_minus_theta = xi * (1 - xi)
            denom = delta + (input_deriv + input_deriv_plus - 2 * delta) * theta_one_minus_theta
            deriv_numer = delta.pow(2) * (
                input_deriv_plus * xi.pow(2) + 2 * delta * theta_one_minus_theta
                + input_deriv * (1 - xi).pow(2)
            )
            logdet = -(torch.log(deriv_numer) - 2 * torch.log(denom))
        else:
            xi = (inp_in - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = xi * (1 - xi)
            numer = input_heights * (delta * xi.pow(2) + input_deriv * theta_one_minus_theta)
            denom = delta + (input_deriv + input_deriv_plus - 2 * delta) * theta_one_minus_theta
            out = input_cumheights + numer / denom
            deriv_numer = delta.pow(2) * (
                input_deriv_plus * xi.pow(2) + 2 * delta * theta_one_minus_theta
                + input_deriv * (1 - xi).pow(2)
            )
            logdet = torch.log(deriv_numer) - 2 * torch.log(denom)

        outputs[inside] = out
        logabsdet[inside] = logdet

    return outputs, logabsdet


class RQSplineCouplingLayer(nn.Module):
    """Rational-quadratic neural spline coupling (Durkan et al. 2019).

    Same half-masking convention as `AffineCouplingLayer`. For each "free"
    dim we predict num_bins widths, num_bins heights and num_bins-1 interior
    derivatives from an MLP conditioned on the masked dims + context.
    Outside [-tail_bound, tail_bound] the map is the identity so the flow
    retains full support on R^D.
    """
    def __init__(self, dim, context_dim, hidden_dim=128, num_bins=8,
                 tail_bound=3.0, mask_type='half'):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1

        out_per_dim = 3 * num_bins - 1
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim * out_per_dim),
        )
        # Zero-init the final layer so the flow starts near identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, context, reverse=False):
        B = x.shape[0]
        masked = x * self.mask
        params = self.net(torch.cat([masked, context], dim=1))
        params = params.view(B, self.dim, 3 * self.num_bins - 1)
        uw = params[..., :self.num_bins]
        uh = params[..., self.num_bins:2 * self.num_bins]
        ud = params[..., 2 * self.num_bins:]

        y_free, logdet_free = _rational_quadratic_spline(
            x, uw, uh, ud, inverse=reverse, tail_bound=self.tail_bound,
        )
        # Free dims are those where mask == 0. Masked dims pass through.
        free = (1 - self.mask).bool()
        y = torch.where(free, y_free, x)
        log_det = (logdet_free * (1 - self.mask)).sum(dim=1)
        if reverse:
            # For inverse pass, log_det returned by caller is not used in our
            # `forward`-only likelihood, but keep sign consistent for symmetry.
            pass
        return y, log_det


def _make_coupling(coupling_type, dim, context_dim, hidden_dim, mask_type,
                   num_bins=8, tail_bound=3.0):
    if coupling_type == 'affine':
        return AffineCouplingLayer(dim=dim, context_dim=context_dim,
                                   hidden_dim=hidden_dim, mask_type=mask_type)
    if coupling_type == 'spline':
        return RQSplineCouplingLayer(dim=dim, context_dim=context_dim,
                                     hidden_dim=hidden_dim, num_bins=num_bins,
                                     tail_bound=tail_bound, mask_type=mask_type)
    raise ValueError(f"Unknown coupling_type: {coupling_type}")


class NormalizingFlow(nn.Module):
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128,
                 coupling_type='affine', spline_num_bins=8, spline_tail_bound=3.0):
        super().__init__()
        self.param_dim = param_dim
        self.context_dim = context_dim
        self.coupling_type = coupling_type

        self.layers = nn.ModuleList([
            _make_coupling(
                coupling_type=coupling_type,
                dim=param_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd',
                num_bins=spline_num_bins,
                tail_bound=spline_tail_bound,
            )
            for i in range(num_layers)
        ])

        self.register_buffer('base_mean', torch.zeros(param_dim))
        self.register_buffer('base_std', torch.ones(param_dim))

    def forward(self, params, context):
        z = params
        log_det_sum = 0

        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det

        # log N(z | base_mean, base_std), summed over dims
        log_prob_base = -0.5 * (
            LOG_2PI + 2.0 * torch.log(self.base_std)
            + ((z - self.base_mean) / self.base_std) ** 2
        ).sum(dim=1)

        return log_prob_base + log_det_sum

    def sample(self, context, num_samples=1):
        batch_size = context.shape[0]
        context_repeated = context.repeat_interleave(num_samples, dim=0)
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)

        return z


# ---- Embedding networks ------------------------------------------------------
# All three consume (N, num_detectors, seq_len) tensors — the raw shape emitted
# by generate_dataset.py — so the two-detector channel axis is preserved.


class SimpleEmbeddingNetwork(nn.Module):
    """
    MLP embedding that respects the detector channel axis.

    Applies a per-channel MLP to each detector, then concatenates the
    per-channel embeddings and projects to `context_dim`. When
    `share_detector_weights=True` the per-channel MLP weights are shared
    across detectors (cheaper, permutation-equivariant between detectors).
    When False each detector gets its own independent MLP — slightly more
    parameters but lets the model learn detector-specific features (useful
    when H1/L1 noise or antenna response differ materially).
    """
    def __init__(self, num_detectors, seq_len, context_dim=64, hidden_dim=128,
                 dropout=0.1, share_detector_weights=True):
        super().__init__()
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.share_detector_weights = share_detector_weights
        h = hidden_dim * 2
        self.per_channel_out_dim = h

        def make_per_channel():
            return nn.Sequential(
                nn.LayerNorm(seq_len),
                nn.Linear(seq_len, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if share_detector_weights:
            self.per_channel = make_per_channel()
        else:
            self.per_channel = nn.ModuleList([make_per_channel() for _ in range(num_detectors)])

        self.merge = nn.Sequential(
            nn.Linear(num_detectors * h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, context_dim),
        )

    def forward(self, data):
        # data: (N, D, T)
        N, D, T = data.shape
        if self.share_detector_weights:
            per = self.per_channel(data.reshape(N * D, T))  # (N*D, h)
            merged_input = per.reshape(N, D * per.shape[-1])
        else:
            per_channel_outs = [net(data[:, d, :]) for d, net in enumerate(self.per_channel)]
            merged_input = torch.cat(per_channel_outs, dim=1)  # (N, D*h)
        return self.merge(merged_input)


class Conv1DEmbeddingNetwork(nn.Module):
    """1D CNN over detector strains concatenated along the time axis.

    Input (N, D, T) is reshaped to (N, 1, D*T) so convolutions operate on one
    long single-channel sequence (e.g. 2 * 8192 = 16384 samples) rather than
    receiving detectors as separate input channels.
    """
    def __init__(self, num_detectors, seq_len, context_dim=128, num_filters=None,
                 dropout=0.1):
        super().__init__()
        if num_filters is None:
            num_filters = [64, 128, 256]
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.context_dim = context_dim

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )

        in_channels = [1] + list(num_filters[:-1])
        self.conv_stack = nn.Sequential(
            *[block(ic, oc) for ic, oc in zip(in_channels, num_filters)]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, data):
        # data: (N, D, T) -> (N, 1, D*T)
        N, D, T = data.shape
        x = data.reshape(N, 1, D * T)
        x = self.conv_stack(x)
        x = self.global_pool(x).view(N, -1)
        return self.fc(x)


class LSTMEmbeddingNetwork(nn.Module):
    """Per-detector 1-D conv down-sampler → shared BiLSTM → merge.

    Each detector's whitened strain (N, 1, T) passes through a stride-2 conv
    stack that shrinks 8192 → ~1024 steps at `conv_channels[-1]` features.
    Both detectors are then fed through THE SAME BiLSTM (shared weights).
    The two detectors' final hidden states are concatenated and projected
    to `context_dim`.
    """
    def __init__(self, num_detectors, seq_len, context_dim=128, hidden_dim=128,
                 num_layers=2, conv_channels=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        layers, in_c = [], 1
        for out_c in conv_channels:
            layers += [
                nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
            ]
            in_c = out_c
        self.conv_front = nn.Sequential(*layers)
        self._feat_dim = conv_channels[-1]

        self.lstm = nn.LSTM(
            input_size=self._feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        merge_in = num_detectors * hidden_dim * 2
        self.output_proj = nn.Sequential(
            nn.Linear(merge_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, data):
        # data: (N, D, T)
        N, D, T = data.shape
        x = data.reshape(N * D, 1, T)
        x = self.conv_front(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = h.reshape(N, D * h.shape[-1])
        return self.output_proj(h)


class DINGOModel(nn.Module):
    """observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)

    Embedding-specific knobs (optional; used only when the matching
    ``embedding_type`` is selected):
      * ``lstm_hidden_dim``, ``lstm_num_layers`` — BiLSTM width / depth.
      * ``conv1d_num_filters`` — list of output channels for the Conv1D stack.
    These default to the values that were previously hard-coded, so existing
    checkpoints load unchanged.
    """
    def __init__(self, num_detectors, seq_len, param_dim=1, context_dim=64,
                 num_flow_layers=6, hidden_dim=128, embedding_type='simple',
                 embedding_dropout=0.1, share_detector_weights=True,
                 lstm_hidden_dim=128, lstm_num_layers=2,
                 conv1d_num_filters=(64, 128, 256),
                 coupling_type='affine', spline_num_bins=8,
                 spline_tail_bound=3.0):
        super().__init__()

        self.embedding_type = embedding_type
        self.num_detectors = num_detectors
        self.seq_len = seq_len

        if embedding_type == 'lstm':
            self.embedding_net = LSTMEmbeddingNetwork(
                num_detectors=num_detectors, seq_len=seq_len, context_dim=context_dim,
                hidden_dim=lstm_hidden_dim, num_layers=lstm_num_layers,
                dropout=embedding_dropout,
            )
        elif embedding_type == 'conv1d':
            self.embedding_net = Conv1DEmbeddingNetwork(
                num_detectors=num_detectors, seq_len=seq_len, context_dim=context_dim,
                num_filters=list(conv1d_num_filters),
            )
        elif embedding_type == 'simple':
            self.embedding_net = SimpleEmbeddingNetwork(
                num_detectors=num_detectors, seq_len=seq_len, context_dim=context_dim,
                hidden_dim=hidden_dim, dropout=embedding_dropout,
                share_detector_weights=share_detector_weights,
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=context_dim,
            num_layers=num_flow_layers,
            hidden_dim=hidden_dim,
            coupling_type=coupling_type,
            spline_num_bins=spline_num_bins,
            spline_tail_bound=spline_tail_bound,
        )

    def forward(self, params, data):
        context = self.embedding_net(data)
        return self.flow(params, context)

    def sample_posterior(self, data, num_samples=1000):
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples


# ==============================================================================
# DATASET LOADING (bridges generate_dataset.py -> training)
# ==============================================================================

def crop_to_merger(data, half_width):
    """Centre-crop a (..., T) strain tensor to ±half_width samples around T//2.

    The data-gen pipeline fixes the merger at sample index T//2, so this
    returns `data[..., T//2 - half_width : T//2 + half_width]`. Pass
    `half_width=None` to leave the tensor unchanged.
    """
    if half_width is None:
        return data
    T = data.shape[-1]
    c = T // 2
    lo, hi = c - half_width, c + half_width
    if lo < 0 or hi > T:
        raise ValueError(
            f"crop half_width={half_width} out of bounds for T={T} "
            f"(would need samples [{lo}:{hi}])"
        )
    return data[..., lo:hi].contiguous()


def _m1m2_to_mcq(m1, m2):
    """Return (chirp_mass, mass_ratio q = m_small / m_large in (0, 1])."""
    m_big, m_small = torch.maximum(m1, m2), torch.minimum(m1, m2)
    q = m_small / m_big
    mc = (m1 * m2).pow(0.6) / (m1 + m2).pow(0.2)
    return mc, q


def _mcq_to_m1m2(mc, q):
    """Inverse of _m1m2_to_mcq. Returns (m1, m2) with m1 >= m2."""
    eta = q / (1.0 + q).pow(2)
    M = mc / eta.pow(0.6)
    m1 = M / (1.0 + q)
    m2 = M * q / (1.0 + q)
    return m1, m2


def resolve_crop_for_embedding(cfg):
    """Select the crop half-width to apply given the active embedding.

    Per-embedding overrides (conv1d_crop_half_width, simple_crop_half_width,
    lstm_crop_half_width) take precedence over the base merger_crop_half_width
    if set (non-None). Makes it natural to HP-scan window size per embedding
    in `hp_search.py` — e.g. conv1d / simple can use a tight 100-sample crop
    while lstm keeps the full 500.
    """
    base = cfg.get('merger_crop_half_width')
    key = f"{cfg.get('embedding_type')}_crop_half_width"
    override = cfg.get(key)
    return override if override is not None else base


def load_dataset_pt(path, use_whitened=True, merger_crop_half_width=None,
                    param_parameterization='m1_m2'):
    """Load dataset.pt from generate_dataset.py, split it, and z-score the params.

    Keeps the detector channel axis: data tensors have shape (N, num_detectors, T).
    If `merger_crop_half_width` is an int, strain is centre-cropped to
    ±half_width samples around the merger (which the data-gen pipeline
    places at T//2), giving a new time axis of 2*half_width.

    When ``param_parameterization='Mc_q'``, the ``mass1`` and ``mass2`` columns
    are replaced with ``chirp_mass`` and ``mass_ratio`` (q = m_small/m_large).
    Training, val, and test splits all share the same transform; the flow
    learns posteriors directly in (Mc, q) space.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run HPC/Pipeline/Data Generation/generate_dataset.py "
            f"and copy/move the produced dataset.pt next to this script."
        )

    raw = torch.load(path, weights_only=False)

    # New datasets (post "saving only processed waveforms now") omit the raw
    # 'X' key — they only store 'X_whitened'. Older dumps have both. We prefer
    # whitened when available, and raise a clear error if raw strain is
    # requested from a processed-only dump.
    X_key = 'X_whitened' if use_whitened else 'X'
    if X_key not in raw:
        if X_key == 'X' and 'X_whitened' in raw:
            raise KeyError(
                f"'{path}' contains only 'X_whitened' (new data-gen format). "
                f"Re-run generate_dataset.py to include raw 'X' if you need "
                f"use_whitened=False, or set use_whitened=True."
            )
        raise KeyError(f"'{path}' is missing required key '{X_key}'. "
                       f"Found keys: {sorted(raw.keys())}")
    X = raw[X_key].float()      # (N, num_detectors, T)
    X = crop_to_merger(X, merger_crop_half_width)
    y = raw['y'].float()        # (N, P)
    metadata = raw['metadata']
    param_names = list(metadata['parameter_names'])

    # Keep a copy of the raw distance column for optional importance-weighting
    # under a volume-uniform prior. Done here so run_training can pull it
    # without re-loading the dataset.
    distance_col_idx = param_names.index('distance') if 'distance' in param_names else None
    raw_distance = y[:, distance_col_idx].clone() if distance_col_idx is not None else None

    # Reparameterise masses if requested. Affects label dim layout and the
    # param_names list; everything downstream (z-score, flow training,
    # evaluators) reads param_names so the swap is transparent.
    if param_parameterization == 'Mc_q':
        try:
            i1 = param_names.index('mass1')
            i2 = param_names.index('mass2')
        except ValueError as exc:
            raise ValueError("param_parameterization='Mc_q' requires "
                             "'mass1' and 'mass2' in the dataset labels.") from exc
        mc, q = _m1m2_to_mcq(y[:, i1], y[:, i2])
        y[:, i1] = mc
        y[:, i2] = q
        param_names[i1] = 'chirp_mass'
        param_names[i2] = 'mass_ratio'
    elif param_parameterization not in ('m1_m2',):
        raise ValueError(f"Unknown param_parameterization: {param_parameterization!r}")

    train_idx = torch.as_tensor(raw['train_indices'], dtype=torch.long)
    val_idx   = torch.as_tensor(raw['val_indices'],   dtype=torch.long)
    test_idx  = torch.as_tensor(raw['test_indices'],  dtype=torch.long)

    train_X, val_X, test_X = X[train_idx], X[val_idx], X[test_idx]
    train_y_raw, val_y_raw, test_y_raw = y[train_idx], y[val_idx], y[test_idx]

    # z-score using TRAIN split only (no val/test leakage).
    means = train_y_raw.mean(dim=0)
    stds  = train_y_raw.std(dim=0)
    safe_stds = torch.where(stds > 0, stds, torch.ones_like(stds))

    def znorm(t):
        out = (t - means) / safe_stds
        out[:, stds == 0] = 0.0
        return out

    train_y = znorm(train_y_raw)
    val_y   = znorm(val_y_raw)
    test_y  = znorm(test_y_raw)

    param_norm_info = {
        name: {
            'mean':   float(means[j]),
            'std':    float(stds[j]),
            'min':    float(y[:, j].min()),
            'max':    float(y[:, j].max()),
            'method': 'zscore',
        }
        for j, name in enumerate(param_names)
    }

    # Split the raw distance vector the same way so run_training can build
    # per-split importance weights without re-loading.
    train_distance = raw_distance[train_idx] if raw_distance is not None else None
    val_distance   = raw_distance[val_idx]   if raw_distance is not None else None
    test_distance  = raw_distance[test_idx]  if raw_distance is not None else None

    return {
        'train_data':   train_X, 'train_params': train_y,
        'val_data':     val_X,   'val_params':   val_y,
        'test_data':    test_X,  'test_params':  test_y,
        'param_norm_info': param_norm_info,
        'param_names': param_names,
        'metadata':    metadata,
        'train_distance': train_distance,
        'val_distance':   val_distance,
        'test_distance':  test_distance,
        'param_parameterization': param_parameterization,
    }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_dingo_model(model, train_params, train_data,
                      num_epochs=20, batch_size=32, lr=1e-4,
                      weight_decay=1e-4,
                      val_params=None, val_data=None,
                      train_weights=None,
                      extra_patience_after_scheduler=3,
                      checkpoint_path=None,
                      checkpoint_extras=None,
                      optimizer_state_dict=None,
                      scheduler_state_dict=None,
                      start_epoch=0,
                      best_log_prob_init=None,
                      best_state_init=None,
                      best_epoch_init=0,
                      bad_epochs_init=0):
    """Train with early stopping + mid-run best-checkpoint persistence.

    Uses AdamW + ReduceLROnPlateau (keyed on val log-prob, falls back to
    train log-prob when no val set is provided).

    Early-stop rule: training halts once validation has failed to improve for
    `scheduler.patience + extra_patience_after_scheduler` epochs in a row —
    i.e. three epochs past the point at which the LR scheduler gave up. With
    defaults (scheduler patience 3, extra 3) that is six bad epochs.

    If `checkpoint_path` is provided, a full checkpoint is written to disk
    every time validation improves, so a cancelled run still leaves the best
    model on disk. `checkpoint_extras` is a dict whose entries are merged
    into each saved checkpoint (e.g. config, param_norm_info).

    Pass the *_state_dict and *_init arguments to resume training.

    Returns (losses, val_losses, best_state_dict, best_log_prob, best_epoch,
             optimizer, scheduler, bad_epochs).
    """
    if torch.isnan(train_params).any() or torch.isnan(train_data).any():
        raise ValueError("Input data contains NaN values")

    print(f"\nData stats:")
    print(f"  train_params: min={train_params.min():.4f}, max={train_params.max():.4f}, "
          f"mean={train_params.mean():.4f}, std={train_params.std():.4f}")
    print(f"  train_data:   min={train_data.min():.4f}, max={train_data.max():.4f}, "
          f"mean={train_data.mean():.4f}, std={train_data.std():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=lr * 0.01
    )
    patience = scheduler.patience + extra_patience_after_scheduler

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    num_samples = len(train_params)
    has_val = val_params is not None and val_data is not None

    print(f"\nTraining for up to {num_epochs} epochs")
    print(f"  Early-stop patience: {patience}  "
          f"(scheduler.patience={scheduler.patience} + extra={extra_patience_after_scheduler})")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    if checkpoint_path is not None:
        print(f"  Persisting best model to disk on each improvement: {checkpoint_path}")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")
    print()

    losses = []
    val_losses = []

    best_log_prob = -float('inf') if best_log_prob_init is None else best_log_prob_init
    best_state = (copy.deepcopy(best_state_init) if best_state_init is not None
                  else copy.deepcopy(model.state_dict()))
    best_epoch = best_epoch_init
    bad_epochs = bad_epochs_init

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_log_prob = 0.0
        num_batches = 0
        batch_log_probs = []

        indices = torch.randperm(num_samples)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch_params = train_params[batch_indices]
            batch_data = train_data[batch_indices]

            if batch_params.shape[0] < 2:
                continue

            optimizer.zero_grad()

            log_prob = model(batch_params, batch_data)
            if train_weights is not None:
                w = train_weights[batch_indices]
                loss = -(w * log_prob).mean()
            else:
                loss = -log_prob.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN/Inf loss at epoch {epoch+1}, batch {i//batch_size + 1} — skipped")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lp = -loss.item()
            epoch_log_prob += lp
            batch_log_probs.append(lp)
            num_batches += 1

        avg_log_prob = epoch_log_prob / num_batches if num_batches > 0 else float('nan')
        losses.append(avg_log_prob)

        avg_val_log_prob = float('nan')
        if has_val:
            model.eval()
            with torch.inference_mode():
                vlog_prob = model(val_params, val_data)
                if not (torch.isnan(vlog_prob).any() or torch.isinf(vlog_prob).any()):
                    avg_val_log_prob = vlog_prob.mean().item()
            val_losses.append(avg_val_log_prob)

        # Track best model on validation log-prob (fallback to train).
        current_metric = avg_val_log_prob if has_val and not math.isnan(avg_val_log_prob) else avg_log_prob

        if not math.isnan(current_metric):
            scheduler.step(current_metric)

        if not math.isnan(current_metric) and current_metric > best_log_prob:
            best_log_prob = current_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad_epochs = 0
            marker = " *"
            if checkpoint_path is not None:
                ckpt = {
                    'model_state_dict': best_state,
                    'best_state': best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': list(losses),
                    'val_losses': list(val_losses),
                    'best_log_prob': best_log_prob,
                    'best_epoch': best_epoch,
                    'bad_epochs': bad_epochs,
                    'epochs_completed': epoch + 1,
                }
                if checkpoint_extras:
                    ckpt.update(checkpoint_extras)
                torch.save(ckpt, checkpoint_path)
        else:
            bad_epochs += 1
            marker = ""

        current_lr = optimizer.param_groups[0]['lr']
        batch_std = float(np.std(batch_log_probs)) if batch_log_probs else float('nan')
        val_str = f", Val: {avg_val_log_prob:7.4f}" if has_val else ""
        print(f"Epoch {epoch+1:3d}/{num_epochs}, Train: {avg_log_prob:7.4f}{val_str}, "
              f"Best: {best_log_prob:7.4f} @ ep{best_epoch}, Std: {batch_std:6.4f}, LR: {current_lr:.2e}{marker}")

        if bad_epochs >= patience:
            print(f"\nEarly stop: no improvement for {patience} epochs "
                  f"(scheduler.patience={scheduler.patience} + {extra_patience_after_scheduler}; "
                  f"best was epoch {best_epoch}).")
            break

    # Restore best weights so the caller sees the best model.
    model.load_state_dict(best_state)
    print(f"\nTraining complete. Best log-prob: {best_log_prob:.4f} (epoch {best_epoch}).")
    return losses, val_losses, best_state, best_log_prob, best_epoch, optimizer, scheduler, bad_epochs


# ==============================================================================
# CONFIG-DRIVEN ENTRY POINT
# ==============================================================================

# All config keys accepted by `run_training`. Anything not present in the
# caller's config falls back to these defaults. Keep this flat so the HP
# search can sweep individual keys without knowing the nested schema.
DEFAULT_CONFIG = {
    # Data
    'dataset_path':             'Data/dataset.pt',
    'use_whitened':             True,
    # ±N samples around the merger (index T//2); None keeps full window.
    'merger_crop_half_width':   500,
    # Per-embedding crop overrides. If set (not None), they replace
    # merger_crop_half_width for the matching embedding. Lets hp_search
    # sweep window size per-embedding (e.g. tight window for conv1d/simple,
    # longer for lstm).
    'simple_crop_half_width':   None,
    'conv1d_crop_half_width':   None,
    'lstm_crop_half_width':     None,
    # Parameter space. 'm1_m2' (default) trains on the 13 raw labels;
    # 'Mc_q' replaces mass1/mass2 with chirp_mass and mass_ratio.
    'param_parameterization':   'm1_m2',
    # 'uniform' (default) → no reweighting; 'volume' → importance-weight
    # each sample by d^2 (normalised) to emulate training under a
    # uniform-in-comoving-volume distance prior with the existing dataset.
    'distance_prior':           'uniform',

    # Model — top-level knobs the HP search is expected to vary
    'embedding_type':           'lstm',      # 'simple' | 'conv1d' | 'lstm'
    'context_dim':              128,
    'num_flow_layers':          4,
    'hidden_dim':               64,
    'embedding_dropout':        0.1,
    'share_detector_weights':   True,
    # Embedding-specific knobs (only used when the matching embedding is picked)
    'lstm_hidden_dim':          128,
    'lstm_num_layers':          2,
    'conv1d_num_filters':       (64, 128, 256),
    # Flow coupling. 'affine' = simple scale+shift (faster, less expressive).
    # 'spline' = rational-quadratic neural spline flow (Durkan 2019; much
    # more expressive — 10x parameter-for-parameter for tight posteriors).
    'coupling_type':            'affine',
    'spline_num_bins':          8,
    'spline_tail_bound':        3.0,

    # Training
    'num_epochs':                   20,
    'batch_size':                   32,
    'learning_rate':                1e-4,
    'weight_decay':                 1e-4,
    # Halt `extra_patience_after_scheduler` epochs after LR scheduler (patience 3)
    # gives up — so default total patience is 6 bad epochs in a row.
    'extra_patience_after_scheduler': 3,

    # Output
    'checkpoint_dir':           '.',
    'device_tag':               'cpu',       # appended to checkpoint filename
    'checkpoint_tag':           '',          # extra suffix (e.g. HP-run id)
    'resume_from':              None,
    'seed':                     0,
}


def _seed_everything(seed):
    """Reseed Python, NumPy, and Torch. Override in GPU version for cuda seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_checkpoint_path(cfg, num_training_samples, add_noise):
    """Compute the canonical save path from a config dict.

    Uses the embedding-resolved crop so per-embedding overrides are visible
    in the filename (e.g. conv1d with simple_crop_half_width=100 writes
    `...conv1d_crop100...`, not the base `crop500`).
    """
    samples_str = (f"{num_training_samples // 1000}k"
                   if num_training_samples >= 1000 else str(num_training_samples))
    if cfg['use_whitened']:
        signal_tag = 'whitened'
    else:
        signal_tag = 'noisy' if add_noise else 'clean'
    active_crop = resolve_crop_for_embedding(cfg)
    crop_tag = f"_crop{active_crop}" if active_crop is not None else ''
    coupling = cfg.get('coupling_type', 'affine')
    coupling_tag = f"_{coupling}" if coupling != 'affine' else ''
    param_tag = ('_Mcq' if cfg.get('param_parameterization') == 'Mc_q' else '')
    prior_tag = ('_volprior' if cfg.get('distance_prior') == 'volume' else '')
    tag = f"_{cfg['checkpoint_tag']}" if cfg['checkpoint_tag'] else ''
    device_tag = f"_{cfg['device_tag']}" if cfg['device_tag'] else ''
    fname = (f"dingo_N{samples_str}_F{cfg['num_flow_layers']}_C{cfg['context_dim']}"
             f"_H{cfg['hidden_dim']}_E{cfg['num_epochs']}_{cfg['embedding_type']}"
             f"{coupling_tag}{crop_tag}{param_tag}{prior_tag}_{signal_tag}{tag}{device_tag}.pt")
    return os.path.join(cfg['checkpoint_dir'], fname)


def run_training(config=None, *, device=None, seed_fn=None):
    """Run one training trial and return a summary dict.

    Parameters
    ----------
    config : dict | None
        Any subset of `DEFAULT_CONFIG`. Missing keys fall back to defaults.
        Call sites can keep passing flat dicts — this is the contract the
        HP search relies on.
    device : torch.device | None
        Override the module-level DEVICE (used by the GPU variant).
    seed_fn : callable | None
        Callable taking an int seed. Defaults to the CPU-safe
        `_seed_everything`; GPU variant passes a cuda-aware one.

    Returns
    -------
    dict
        {best_log_prob, best_epoch, epochs_completed, num_params,
         elapsed_sec, checkpoint_path, config}
    """
    import time
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    dev = device if device is not None else DEVICE
    (seed_fn or _seed_everything)(cfg['seed'])

    # ----- Load data -----
    print(f"\nLoading dataset from: {cfg['dataset_path']}")
    active_crop = resolve_crop_for_embedding(cfg)
    if active_crop is not None:
        print(f"  Cropping strain to ±{active_crop} samples around merger "
              f"(new T = {2 * active_crop}; embedding={cfg['embedding_type']})")
    print(f"  Parameterisation: {cfg['param_parameterization']}")
    print(f"  Distance prior:   {cfg['distance_prior']}")
    ds = load_dataset_pt(cfg['dataset_path'],
                         use_whitened=cfg['use_whitened'],
                         merger_crop_half_width=active_crop,
                         param_parameterization=cfg['param_parameterization'])

    train_data,   train_params = ds['train_data'],   ds['train_params']
    val_data,     val_params   = ds['val_data'],     ds['val_params']
    test_data,    test_params  = ds['test_data'],    ds['test_params']
    param_norm_info = ds['param_norm_info']
    param_names     = ds['param_names']
    metadata        = ds['metadata']

    param_dim = len(param_names)
    _, num_detectors, seq_len = train_data.shape
    num_training_samples = len(train_params)
    add_noise = metadata.get('add_noise', True)

    print(f"  Train:      {len(train_params)}")
    print(f"  Validation: {len(val_params)}")
    print(f"  Test:       {len(test_params)}")
    print(f"  Data shape: (N, {num_detectors}, {seq_len})  "
          f"({'whitened' if cfg['use_whitened'] else 'raw'})")
    print(f"  Params:     {param_names}")

    # ----- Save path -----
    save_path = build_checkpoint_path(cfg, num_training_samples, add_noise)
    print(f"\nModel will be saved as: {save_path}")

    # ----- Build model -----
    print("\nModel:")
    print(f"  PARAM_DIM={param_dim}  NUM_DETECTORS={num_detectors}  SEQ_LEN={seq_len}")
    print(f"  CONTEXT_DIM={cfg['context_dim']}  NUM_FLOW_LAYERS={cfg['num_flow_layers']}  "
          f"HIDDEN_DIM={cfg['hidden_dim']}  EMBEDDING={cfg['embedding_type']}")

    model = DINGOModel(
        num_detectors=num_detectors,
        seq_len=seq_len,
        param_dim=param_dim,
        context_dim=cfg['context_dim'],
        num_flow_layers=cfg['num_flow_layers'],
        hidden_dim=cfg['hidden_dim'],
        embedding_type=cfg['embedding_type'],
        embedding_dropout=cfg['embedding_dropout'],
        share_detector_weights=cfg['share_detector_weights'],
        lstm_hidden_dim=cfg['lstm_hidden_dim'],
        lstm_num_layers=cfg['lstm_num_layers'],
        conv1d_num_filters=cfg['conv1d_num_filters'],
        coupling_type=cfg['coupling_type'],
        spline_num_bins=cfg['spline_num_bins'],
        spline_tail_bound=cfg['spline_tail_bound'],
    ).to(dev)

    num_params = sum(p.numel() for p in model.parameters())

    # ----- Optional: resume from checkpoint -----
    optimizer_state_dict = None
    scheduler_state_dict = None
    start_epoch = 0
    best_log_prob_init = None
    best_state_init = None
    best_epoch_init = 0
    bad_epochs_init = 0
    if cfg['resume_from'] is not None:
        print(f"\nResuming from checkpoint: {cfg['resume_from']}")
        ckpt = torch.load(cfg['resume_from'], weights_only=False, map_location=dev)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer_state_dict = ckpt.get('optimizer_state_dict')
        scheduler_state_dict = ckpt.get('scheduler_state_dict')
        start_epoch = ckpt.get('epochs_completed', 0)
        best_log_prob_init = ckpt.get('best_log_prob')
        best_state_init = ckpt.get('best_state', ckpt.get('model_state_dict'))
        best_epoch_init = ckpt.get('best_epoch', 0)
        bad_epochs_init = ckpt.get('bad_epochs', 0)
        print(f"  Resumed @ epoch {start_epoch}, best_log_prob={best_log_prob_init}")

    # Metadata merged into every checkpoint written during training. The
    # training loop persists the best model on each improvement, so a
    # killed run still leaves the best state so far on disk.
    checkpoint_extras = {
        'param_norm_info':   param_norm_info,
        'model_param_names': param_names,
        'config': {
            'num_detectors':          num_detectors,
            'seq_len':                seq_len,
            'param_dim':              param_dim,
            'context_dim':            cfg['context_dim'],
            'num_flow_layers':        cfg['num_flow_layers'],
            'hidden_dim':             cfg['hidden_dim'],
            'embedding_type':         cfg['embedding_type'],
            'embedding_dropout':      cfg['embedding_dropout'],
            'share_detector_weights': cfg['share_detector_weights'],
            'lstm_hidden_dim':        cfg['lstm_hidden_dim'],
            'lstm_num_layers':        cfg['lstm_num_layers'],
            'conv1d_num_filters':     list(cfg['conv1d_num_filters']),
            'coupling_type':          cfg['coupling_type'],
            'spline_num_bins':        cfg['spline_num_bins'],
            'spline_tail_bound':      cfg['spline_tail_bound'],
            'num_epochs':             cfg['num_epochs'],
            'batch_size':             cfg['batch_size'],
            'learning_rate':          cfg['learning_rate'],
            'weight_decay':           cfg['weight_decay'],
            'num_training_samples':   num_training_samples,
            'add_noise':              add_noise,
            'whiten':                 cfg['use_whitened'],
            'merger_crop_half_width': active_crop,
            'simple_crop_half_width': cfg['simple_crop_half_width'],
            'conv1d_crop_half_width': cfg['conv1d_crop_half_width'],
            'lstm_crop_half_width':   cfg['lstm_crop_half_width'],
            'param_parameterization': cfg['param_parameterization'],
            'distance_prior':         cfg['distance_prior'],
            'seed':                   cfg['seed'],
        },
    }

    # ----- Distance importance weights for a volume-uniform prior -----
    # Dataset samples distance uniformly in [D_min, D_max]; the BBH prior
    # is uniform-in-comoving-volume ∝ D^2. Reweight loss by normalised D^2
    # so the flow learns posteriors under the physical prior without
    # regenerating the dataset.
    train_weights = None
    if cfg['distance_prior'] == 'volume':
        if ds.get('train_distance') is None:
            raise ValueError("distance_prior='volume' requires 'distance' "
                             "in the dataset labels.")
        d = ds['train_distance'].to(dev)
        w = d.pow(2)
        train_weights = w / w.mean()  # normalise so mean weight = 1
        print(f"  Volume-prior weights: min={train_weights.min():.3f}, "
              f"max={train_weights.max():.3f}")
    elif cfg['distance_prior'] not in ('uniform',):
        raise ValueError(f"Unknown distance_prior: {cfg['distance_prior']!r}")

    # ----- Train -----
    t0 = time.time()
    (losses, val_losses, best_state, best_log_prob, best_epoch,
     optimizer, scheduler, bad_epochs) = train_dingo_model(
        model, train_params, train_data,
        num_epochs=cfg['num_epochs'], batch_size=cfg['batch_size'],
        lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
        val_params=val_params, val_data=val_data,
        train_weights=train_weights,
        extra_patience_after_scheduler=cfg['extra_patience_after_scheduler'],
        checkpoint_path=save_path,
        checkpoint_extras=checkpoint_extras,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        start_epoch=start_epoch,
        best_log_prob_init=best_log_prob_init,
        best_state_init=best_state_init,
        best_epoch_init=best_epoch_init,
        bad_epochs_init=bad_epochs_init,
    )
    elapsed = time.time() - t0

    print(f"\nBest model (epoch {best_epoch}) saved to: {save_path}")
    print(f"Total parameters: {num_params:,}   Wall-clock: {elapsed:.1f}s")

    return {
        'best_log_prob':     float(best_log_prob),
        'best_epoch':        int(best_epoch),
        'epochs_completed':  start_epoch + len(losses),
        'num_params':        int(num_params),
        'elapsed_sec':       float(elapsed),
        'checkpoint_path':   save_path,
        'config':            cfg,
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    # Edit here to run a single training job. For a sweep across model
    # types / sizes / depths, use hp_search.py instead.
    run_training({
        'dataset_path':           'Data/dataset.pt',
        'merger_crop_half_width': 500,
        'embedding_type':         'lstm',
        'context_dim':            128,
        'num_flow_layers':        4,
        'hidden_dim':             64,
        'num_epochs':             20,
        'batch_size':             32,
        'learning_rate':          1e-4,
        'device_tag':             'cpu',
    })
