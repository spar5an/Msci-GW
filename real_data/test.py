from gwosc import datasets
import requests
from pesummary.io import read
from pesummary.gw.fetch import fetch_open_samples, download_and_read_file
import pandas as pd
import h5py
import numpy as np
from scipy.stats import gaussian_kde




events = datasets.find_datasets()

both_capture = []

for event in events:
    try:
        if datasets.event_detectors(event) == {'L1', 'H1'}:
            both_capture.append(event)
            print(event, "found")
    except:
        print("URL not found for event: ", event)
        
print(both_capture)

# Deduplicate: strip version suffixes and keep only GW-named events
unique_events = list(dict.fromkeys(
    e.split("-")[0] for e in both_capture if e.startswith("GW")
))
print(f"\n{len(unique_events)} unique GW events in both H1 & L1")

def kde_mode(samples):
    kde = gaussian_kde(samples)
    x = np.linspace(min(samples), max(samples), 1000)
    return x[np.argmax(kde(x))]

params_to_extract = [
    'ra', 'dec', 'psi', 'mass_1_source', 'mass_2_source', 'chirp_mass_source',
    'luminosity_distance', 'chi_eff', 'chi_p',
    'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
    'redshift', 'geocent_time'
]

rows = []
for i, event in enumerate(unique_events):
    print(f"[{i+1}/{len(unique_events)}] Fetching posteriors for {event}...")
    try:
        data = fetch_open_samples(event)
    except Exception:
        # Fallback: look up PE file URL from GWOSC API v2
        try:
            api_url = f"https://gwosc.org/api/v2/event-versions/{event}-v1/parameters"
            params_resp = requests.get(api_url).json()
            pe_url = None
            for p in params_resp.get("results", []):
                url = p.get("data_url")
                if url and url.endswith(".hdf5/content"):
                    pe_url = url
            if pe_url is None:
                print(f"  Skipping {event}: no PE file URL found on GWOSC")
                continue
            print(f"  Fallback: downloading from GWOSC API")
            data = download_and_read_file(pe_url, outdir=".", read_file=True)
        except Exception as e:
            print(f"  Skipping {event}: {e}")
            continue
    try:
        samples = data.samples_dict
        # Handle MultiAnalysis files (GWTC-2+) keyed by analysis label
        if hasattr(samples, 'keys') and not any(p in samples for p in params_to_extract):
            label = list(samples.keys())[0]
            samples = samples[label]
        row = {'event': event}
        try:
            row['gps_time'] = datasets.event_gps(event)
        except Exception:
            row['gps_time'] = None
        for p in params_to_extract:
            row[p] = kde_mode(samples[p]) if p in samples else None
        rows.append(row)
        print(f"  OK — ra={row['ra']}, dec={row['dec']}, gps={row['gps_time']}")
    except Exception as e:
        print(f"  Skipping {event}: {e}")

df = pd.DataFrame(rows)
print(df)
df.to_csv("gw_events_stats.csv", index=False)

#query for waveforms
#download, cut and then move pytorch tensor

#processing