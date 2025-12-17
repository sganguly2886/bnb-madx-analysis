import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Read MAD-X twiss output
def read_twiss(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    header = next(line for line in lines if line.startswith('*')).lstrip('*').split()
    data = [line.split() for line in lines if not line.startswith(('*', '$')) and line.strip()]
    df = pd.DataFrame(data, columns=[h.strip('"') for h in header])
    df = df[['S', 'BETX', 'DX']].apply(pd.to_numeric, errors='coerce')
    return df[(df['S'] >= 195) & (df['S'] <= 215)]  # final focus region

# Color + linestyle map by betx shift
betx_styles = {
    "minus10": ("gray", ":"),
    "minus5": ("dodgerblue", "--"),
    "default": ("black", "-"),
    "plus5": ("orange", "--"),
    "plus10": ("firebrick", ":")
}

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Loop over .out files
for filepath in sorted(glob.glob("twiss_HQ873_874_*.out")):
    filename = os.path.basename(filepath).replace("twiss_HQ873_874_", "").replace(".out", "")
    parts = filename.split("_betx")
    k1_variation = parts[0]
    betx_shift = parts[1] if len(parts) > 1 else "default"

    if betx_shift not in betx_styles:
        continue

    color, linestyle = betx_styles[betx_shift]
    label = f"{k1_variation}, βx {betx_shift}"

    df = read_twiss(filepath)
    ax1.plot(df['S'], df['BETX'], label=label, color=color, linestyle=linestyle, linewidth=2)
    ax2.plot(df['S'], df['DX'], color=color, linestyle=linestyle, linewidth=2)

# Format
ax1.set_ylabel("BETX (m)")
ax1.set_title("Zoom on BETX and DX for HQ873+HQ874 ±10% and Twiss BETX Variations")
ax1.grid(True)
ax2.set_xlabel("S (m)")
ax2.set_ylabel("DX (m)")
ax2.grid(True)

# Legend
ax1.legend(title="K1, Twiss Variation", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize='small')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("betx_dx_HQ873_874_joint_k1_betx_variation_full.png", dpi=300)
plt.show()
