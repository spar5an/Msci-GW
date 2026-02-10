"""
Real Gravitational Wave Data Module

Provides utilities for loading, processing, and comparing real GW events
from GWTC catalogs with simulated data.

Usage
-----
>>> from real_data import (
...     get_available_events,
...     get_event_parameters,
...     load_real_event,
...     create_real_data_dataloaders,
...     compare_real_vs_simulated,
...     generate_comparison_waveform
... )

>>> # List available events
>>> events = get_available_events('gwtc-3')
>>> print(f"Found {len(events)} events")

>>> # Get parameters for a specific event
>>> params = get_event_parameters('GW150914')
>>> print(f"Masses: {params['mass1']:.1f}, {params['mass2']:.1f} Msun")

>>> # Load real data as JHPY-compatible DataLoaders
>>> result = create_real_data_dataloaders(['GW150914', 'GW151226'])

>>> # Compare with simulated waveform
>>> simulated = generate_comparison_waveform('GW150914')
>>> comparison = compare_real_vs_simulated(result, simulated)
>>> print(f"Match: {comparison['match_values']['H1']:.3f}")
"""

try:
    from .catalog_utils import (
        get_available_events,
        get_event_parameters,
        get_event_metadata,
        filter_events_by_criteria,
        parameters_to_jhpy_format,
        get_event_summary
    )

    from .real_data_loader import (
        load_real_event,
        load_multiple_events,
        extract_signal_window,
        create_real_data_dataloaders,
        save_real_dataloaders,
        load_real_dataloaders
    )

    from .event_processor import (
        notch_filter_powerlines,
        estimate_psd_from_offsource,
        whiten_real_strain,
        check_data_quality,
        interpolate_gaps,
        process_real_strain,
        apply_real_data_processing_pipeline
    )

    from .comparison_utils import (
        generate_comparison_waveform,
        compute_match,
        compute_snr,
        compute_residual,
        compare_real_vs_simulated,
        create_comparison_plot,
        batch_compare_events,
        compute_parameter_recovery_accuracy,
        print_comparison_summary
    )
except ImportError:
    from catalog_utils import (
        get_available_events,
        get_event_parameters,
        get_event_metadata,
        filter_events_by_criteria,
        parameters_to_jhpy_format,
        get_event_summary
    )

    from real_data_loader import (
        load_real_event,
        load_multiple_events,
        extract_signal_window,
        create_real_data_dataloaders,
        save_real_dataloaders,
        load_real_dataloaders
    )

    from event_processor import (
        notch_filter_powerlines,
        estimate_psd_from_offsource,
        whiten_real_strain,
        check_data_quality,
        interpolate_gaps,
        process_real_strain,
        apply_real_data_processing_pipeline
    )

    from comparison_utils import (
        generate_comparison_waveform,
        compute_match,
        compute_snr,
        compute_residual,
        compare_real_vs_simulated,
        create_comparison_plot,
        batch_compare_events,
        compute_parameter_recovery_accuracy,
        print_comparison_summary
    )

__all__ = [
    # Catalog utilities
    'get_available_events',
    'get_event_parameters',
    'get_event_metadata',
    'filter_events_by_criteria',
    'parameters_to_jhpy_format',
    'get_event_summary',

    # Data loading
    'load_real_event',
    'load_multiple_events',
    'extract_signal_window',
    'create_real_data_dataloaders',
    'save_real_dataloaders',
    'load_real_dataloaders',

    # Signal processing
    'notch_filter_powerlines',
    'estimate_psd_from_offsource',
    'whiten_real_strain',
    'check_data_quality',
    'interpolate_gaps',
    'process_real_strain',
    'apply_real_data_processing_pipeline',

    # Comparison utilities
    'generate_comparison_waveform',
    'compute_match',
    'compute_snr',
    'compute_residual',
    'compare_real_vs_simulated',
    'create_comparison_plot',
    'batch_compare_events',
    'compute_parameter_recovery_accuracy',
    'print_comparison_summary'
]

__version__ = '1.0.0'
