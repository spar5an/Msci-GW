"""
Catalog utilities for extracting GW event parameters from GWTC.

This module provides functions to access gravitational wave event catalogs
(GWTC-1, GWTC-2, GWTC-3) via PyCBC and extract physical parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pycbc.catalog import Catalog, Merger


def get_available_events(catalog: str = 'gwtc-3') -> List[str]:
    """
    Get list of all available GW events from specified catalog.

    Parameters
    ----------
    catalog : str
        Catalog name: 'gwtc-1', 'gwtc-2', 'gwtc-2.1', 'gwtc-3'
        Default: 'gwtc-3' (most comprehensive)

    Returns
    -------
    List[str]
        List of event names (e.g., ['GW150914', 'GW151226', ...])

    Examples
    --------
    >>> events = get_available_events('gwtc-1')
    >>> print(events[:5])
    ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608']
    """
    try:
        cat = Catalog(source=catalog)
        return list(cat.names)
    except Exception as e:
        print(f"Warning: Could not access catalog {catalog}: {e}")
        # Fall back to known events
        return ['GW150914', 'GW151226', 'GW170104', 'GW170608',
                'GW170729', 'GW170809', 'GW170814', 'GW170817',
                'GW170818', 'GW170823']


def get_event_parameters(event_name: str) -> Dict:
    """
    Extract physical parameters for a specific GW event.

    Parameters
    ----------
    event_name : str
        Event name (e.g., 'GW150914', 'GW170817')

    Returns
    -------
    Dict containing:
        - mass1: Primary mass (solar masses, detector frame)
        - mass2: Secondary mass (solar masses, detector frame)
        - mass1_source: Source-frame primary mass
        - mass2_source: Source-frame secondary mass
        - chirp_mass: Chirp mass
        - total_mass: Total mass
        - mass_ratio: Mass ratio (q = m2/m1)
        - spin1z: Primary spin z-component (if available)
        - spin2z: Secondary spin z-component (if available)
        - chi_eff: Effective spin parameter
        - luminosity_distance: Distance in Mpc
        - redshift: Cosmological redshift
        - ra: Right ascension (radians)
        - dec: Declination (radians)
        - tc: GPS coalescence time
        - network_snr: Network signal-to-noise ratio
        - far: False alarm rate (per year)
        - event_type: 'BBH', 'BNS', or 'NSBH'
        - parameter_uncertainties: Dict of (lower, upper) bounds

    Examples
    --------
    >>> params = get_event_parameters('GW150914')
    >>> print(f"Masses: {params['mass1']:.1f}, {params['mass2']:.1f} Msun")
    Masses: 35.6, 30.6 Msun
    """
    merger = Merger(event_name)

    params = {}
    uncertainties = {}

    # Get available data keys to avoid triggering pycbc error messages
    available_keys = set(merger.data.keys()) if hasattr(merger, 'data') else set()

    # Helper to safely get median value (only tries if key exists)
    def safe_median(param_name, default=None):
        if param_name not in available_keys:
            return default
        try:
            return merger.median1d(param_name)
        except (KeyError, ValueError, RuntimeError):
            return default

    # Helper to safely get confidence interval from _lower and _upper fields
    def safe_confidence(param_name):
        lower_key = param_name + '_lower'
        upper_key = param_name + '_upper'
        if param_name not in available_keys or lower_key not in available_keys or upper_key not in available_keys:
            return None
        try:
            median_val = merger.median1d(param_name)
            lower_val = merger.median1d(lower_key)
            upper_val = merger.median1d(upper_key)
            if median_val is not None and lower_val is not None and upper_val is not None:
                return (median_val - lower_val, upper_val - median_val)
            return None
        except (KeyError, ValueError, RuntimeError, AttributeError):
            return None

    # Source frame masses (these are what's available in GWTC)
    params['mass1_source'] = safe_median('mass_1_source')
    params['mass2_source'] = safe_median('mass_2_source')

    # Mass parameters - use source frame if detector frame not available
    params['mass1'] = safe_median('mass_1') or params['mass1_source']
    params['mass2'] = safe_median('mass_2') or params['mass2_source']

    # Derived mass parameters
    params['chirp_mass'] = safe_median('chirp_mass')
    params['total_mass'] = safe_median('total_mass') or safe_median('total_mass_source')
    params['mass_ratio'] = safe_median('mass_ratio')

    # Spin parameters - chi_eff is usually available, individual spins often not
    params['chi_eff'] = safe_median('chi_eff', 0.0)
    params['spin1z'] = safe_median('spin_1z', 0.0)
    params['spin2z'] = safe_median('spin_2z', 0.0)

    # Distance and redshift
    params['luminosity_distance'] = safe_median('luminosity_distance')
    params['redshift'] = safe_median('redshift')

    # Sky location (often not available in catalog)
    params['ra'] = safe_median('ra')
    params['dec'] = safe_median('dec')

    # Timing
    try:
        params['tc'] = merger.time
    except (AttributeError, KeyError):
        params['tc'] = safe_median('geocent_time')

    # Detection statistics
    params['network_snr'] = safe_median('network_matched_filter_snr')
    params['far'] = safe_median('far')

    # Determine event type based on masses
    m1 = params.get('mass1_source') or params.get('mass1')
    m2 = params.get('mass2_source') or params.get('mass2')

    if m1 is not None and m2 is not None:
        if m1 < 3 and m2 < 3:
            params['event_type'] = 'BNS'
        elif m1 >= 3 and m2 < 3:
            params['event_type'] = 'NSBH'
        else:
            params['event_type'] = 'BBH'
    else:
        params['event_type'] = 'unknown'

    # Get uncertainties for key parameters
    for param_name in ['mass_1', 'mass_2', 'luminosity_distance', 'chi_eff']:
        ci = safe_confidence(param_name)
        if ci is not None:
            uncertainties[param_name] = ci

    params['parameter_uncertainties'] = uncertainties

    return params


def get_event_metadata(event_name: str) -> Dict:
    """
    Get metadata about a GW event (discovery info, detectors, etc).

    Parameters
    ----------
    event_name : str
        Event name

    Returns
    -------
    Dict containing:
        - gps_time: GPS time of coalescence
        - detectors: List of detectors with data ['H1', 'L1', 'V1']
        - catalog_origin: Which GWTC catalog
        - reference_url: Link to GWOSC
    """
    merger = Merger(event_name)

    metadata = {}

    # GPS time
    try:
        metadata['gps_time'] = merger.time
    except (AttributeError, KeyError):
        metadata['gps_time'] = None

    # Available detectors
    available_detectors = []
    for det in ['H1', 'L1', 'V1', 'K1']:
        try:
            # Check if strain data is available for this detector
            _ = merger.strain(det)
            available_detectors.append(det)
        except Exception:
            pass
    metadata['detectors'] = available_detectors if available_detectors else ['H1', 'L1']

    # Catalog info
    metadata['catalog_origin'] = 'GWTC'
    metadata['reference_url'] = f'https://www.gw-openscience.org/eventapi/html/GWTC/'

    return metadata


def filter_events_by_criteria(
    min_mass: float = None,
    max_mass: float = None,
    min_snr: float = None,
    event_type: str = None,
    detectors: List[str] = None,
    catalog: str = 'gwtc-3'
) -> List[str]:
    """
    Filter catalog events by physical criteria.

    Parameters
    ----------
    min_mass : float, optional
        Minimum total mass (solar masses)
    max_mass : float, optional
        Maximum total mass (solar masses)
    min_snr : float, optional
        Minimum network SNR
    event_type : str, optional
        'BBH', 'BNS', or 'NSBH'
    detectors : List[str], optional
        Required detectors (e.g., ['H1', 'L1'])
    catalog : str
        Catalog to search. Default: 'gwtc-3'

    Returns
    -------
    List[str]
        Filtered event names
    """
    all_events = get_available_events(catalog)
    filtered = []

    for event_name in all_events:
        try:
            params = get_event_parameters(event_name)

            # Check total mass
            total_mass = params.get('total_mass')
            if total_mass is None:
                m1, m2 = params.get('mass1'), params.get('mass2')
                if m1 is not None and m2 is not None:
                    total_mass = m1 + m2

            if min_mass is not None and total_mass is not None:
                if total_mass < min_mass:
                    continue

            if max_mass is not None and total_mass is not None:
                if total_mass > max_mass:
                    continue

            # Check SNR
            if min_snr is not None:
                snr = params.get('network_snr')
                if snr is None or snr < min_snr:
                    continue

            # Check event type
            if event_type is not None:
                if params.get('event_type') != event_type:
                    continue

            # Check detectors
            if detectors is not None:
                metadata = get_event_metadata(event_name)
                available = metadata.get('detectors', [])
                if not all(d in available for d in detectors):
                    continue

            filtered.append(event_name)

        except Exception as e:
            print(f"Warning: Could not process {event_name}: {e}")
            continue

    return filtered


def parameters_to_jhpy_format(
    params: Dict,
    param_names: List[str] = None
) -> np.ndarray:
    """
    Convert catalog parameters to JHPY parameter array format.

    Maps GWTC parameter names to the format used by pycbc_data_generator:
    ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', ...]

    Parameters
    ----------
    params : Dict
        Parameters from get_event_parameters()
    param_names : List[str], optional
        Ordered list of parameter names to extract.
        Default: ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance']

    Returns
    -------
    np.ndarray
        1D array of parameter values in specified order
    """
    if param_names is None:
        param_names = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance']

    # Mapping from JHPY names to catalog names
    name_mapping = {
        'mass1': ['mass1', 'mass_1', 'mass1_source', 'mass_1_source'],
        'mass2': ['mass2', 'mass_2', 'mass2_source', 'mass_2_source'],
        'spin1z': ['spin1z', 'spin_1z', 'a_1'],
        'spin2z': ['spin2z', 'spin_2z', 'a_2'],
        'distance': ['luminosity_distance', 'distance'],
        'inclination': ['inclination', 'iota'],
        'ra': ['ra', 'right_ascension'],
        'dec': ['dec', 'declination'],
        'polarization': ['polarization', 'psi'],
        'chi_eff': ['chi_eff'],
    }

    values = []
    for name in param_names:
        value = None

        # Try direct lookup
        if name in params and params[name] is not None:
            value = params[name]
        else:
            # Try mapped names
            for mapped_name in name_mapping.get(name, []):
                if mapped_name in params and params[mapped_name] is not None:
                    value = params[mapped_name]
                    break

        # Use default if not found
        if value is None:
            defaults = {
                'spin1z': 0.0,
                'spin2z': 0.0,
                'inclination': 0.0,
                'polarization': 0.0,
            }
            value = defaults.get(name, 0.0)

        values.append(value)

    return np.array(values, dtype=np.float32)


def get_event_summary(event_name: str) -> str:
    """
    Get a human-readable summary of an event's parameters.

    Parameters
    ----------
    event_name : str
        Event name

    Returns
    -------
    str
        Formatted summary string
    """
    params = get_event_parameters(event_name)

    lines = [f"=== {event_name} ==="]
    lines.append(f"Type: {params.get('event_type', 'unknown')}")

    m1, m2 = params.get('mass1'), params.get('mass2')
    if m1 and m2:
        lines.append(f"Masses: {m1:.1f} + {m2:.1f} = {m1+m2:.1f} Msun")

    dist = params.get('luminosity_distance')
    if dist:
        lines.append(f"Distance: {dist:.0f} Mpc")

    snr = params.get('network_snr')
    if snr:
        lines.append(f"SNR: {snr:.1f}")

    chi_eff = params.get('chi_eff')
    if chi_eff is not None:
        lines.append(f"Effective spin: {chi_eff:.2f}")

    return "\n".join(lines)
