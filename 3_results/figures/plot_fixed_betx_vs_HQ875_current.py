import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def read_twiss_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    header_line = next(line for line in lines if line.startswith('*')).strip().lstrip('*').split()
    data_lines = [line.strip() for line in lines if not line.startswith(('*', '$')) and line.strip()]
    data = [line.split() for line in data_lines]
    df = pd.DataFrame(data, columns=[col.strip('"') for col in header_line])
    df = df[['S', 'BETX']].apply(pd.to_numeric, errors='coerce')
    return df[df['S'] <= 220]

# BETX value to plot
target_betx = "0percentbase"

# Define distinct styles
styles = {
    "default": {"linestyle": "-", "linewidth": 1},
    "10percentup": {"linestyle": "--", "linewidth": 2},
    "10percentdown": {"linestyle": ":", "linewidth": 2},
    "20percentup": {"linestyle": "-.", "linewidth": 2},
    "20percentdown": {"linestyle": (0, (3, 5, 1, 5)), "linewidth": 2}
}

# Find matching files
twiss_files = sorted(glob.glob(f"twiss_IQ875_*{target_betx}.out"))

# Plot
plt.figure(figsize=(12, 6))
for file in twiss_files:
    label = os.path.basename(file).replace("twiss_IQ875_", "").replace(".out", "")
    hq_label = label.split('_')[0]
    df = read_twiss_file(file)

    style = styles.get(hq_label, {"linestyle": "-", "linewidth": 1})  # fallback style
    plt.plot(df['S'], df['BETX'], label=hq_label, linestyle=style["linestyle"], linewidth=style["linewidth"])

plt.title(f"BETX vs S for HQ875 Current Sweep (BETX = {target_betx})")
plt.xlabel("S (m)")
plt.ylabel("BETX (m)")
plt.grid(True)
plt.legend(title="HQ875 Current", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(f"BETX_vs_S_HQ875_{target_betx}_styled.png", dpi=300)
plt.show()
