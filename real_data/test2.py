from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv("gw_events_stats.csv")
out_dir = "strain_data"
os.makedirs(out_dir, exist_ok=True)

successful = []

for idx, row in df.iterrows():
    event = row['event']
    gps = row['geocent_time']
    start = int(gps) - 2
    end = int(gps) + 2

    print(f"[{idx+1}/{len(df)}] Downloading {event} (GPS {gps})...")
    ok = True
    for det in ['H1', 'L1']:
        try:
            ts = TimeSeries.fetch_open_data(det, start, end, sample_rate=4096)
            out_path = os.path.join(out_dir, f"{event}_{det}_4096Hz.hdf5")
            ts.write(out_path, overwrite=True)
            print(f"  {det}: OK ({ts.shape})")
        except Exception as e:
            print(f"  {det}: FAILED — {e}")
            ok = False
    if ok:
        successful.append({'event': event, 'gps': gps})

print(f"\nDownloaded {len(successful)}/{len(df)} events successfully.")

# Plot first 3 events as a quick sanity check
plot_events = successful[:3]
if plot_events:
    fig, axes = plt.subplots(len(plot_events), 1, figsize=(12, 3 * len(plot_events)))
    if len(plot_events) == 1:
        axes = [axes]

    for ax, info in zip(axes, plot_events):
        event = info['event']
        gps = info['gps']
        for det, color in [('H1', 'red'), ('L1', 'blue')]:
            path = os.path.join(out_dir, f"{event}_{det}_4096Hz.hdf5")
            ts = TimeSeries.read(path)
            ax.plot(ts.times.value - gps, ts.value, label=det, color=color, alpha=0.7)
        ax.set_title(event)
        ax.set_xlabel('Time relative to merger (s)')
        ax.set_ylabel('Strain')
        ax.legend()

    plt.tight_layout()
    plt.savefig('waveform_check.png', dpi=150)
    plt.show()
    print("Saved waveform_check.png")
