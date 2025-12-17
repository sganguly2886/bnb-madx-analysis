import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def read_twiss(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    header = next(line for line in lines if line.startswith('*')).lstrip('*').split()
    data = [line.split() for line in lines if not line.startswith(('*', '$')) and line.strip()]
    df = pd.DataFrame(data, columns=[h.strip('"') for h in header])
    df = df[['S', 'BETX', 'DX']].apply(pd.to_numeric, errors='coerce')
    return df[(df['S'] >= 195) & (df['S'] <= 215)]

# Color/style mapping for HQ873+874 scans
color_map = {
    "default": "black",
    "10percentup": "orange",
    "10percentdown": "firebrick"
}
style_map = {
    "default": ":",
    "10percentup": "--",
    "10percentdown": "-"
}

# Set up plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot each HQ873+HQ874 scan
for twiss_file in sorted(glob.glob("twiss_HQ873_874_*.out")):
    label = os.path.basename(twiss_file).replace("twiss_HQ873_874_", "").replace(".out", "")
    df = read_twiss(twiss_file)
    color = color_map.get(label, "gray")
    linestyle = style_map.get(label, "-.")

    ax1.plot(df["S"], df["BETX"], label=label, color=color, linestyle=linestyle, linewidth=2)
    ax2.plot(df["S"], df["DX"], label=label, color=color, linestyle=linestyle, linewidth=2)

# Annotate plots
ax1.set_ylabel("BETX (m)")
ax1.set_title("Zoom on BETX and DX with HQ873+HQ874 ±10% (HQ875 Fixed)")
ax1.grid(True)

ax2.set_xlabel("S (m)")
ax2.set_ylabel("DX (m)")
ax2.grid(True)

ax1.legend(title="Variation", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize='small')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("betx_dx_HQ873_874_joint_variation.png", dpi=300)
plt.show()
