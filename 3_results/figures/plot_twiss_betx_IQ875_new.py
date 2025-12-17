import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import cm
from matplotlib.colors import to_hex

def read_twiss_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_line = next(line for line in lines if line.startswith('*')).strip().lstrip('*').split()
    data_lines = [line.strip() for line in lines if not line.startswith(('*', '$')) and line.strip()]
    data = [line.split() for line in data_lines]

    df = pd.DataFrame(data, columns=[col.strip('"') for col in header_line])
    df = df[['S', 'BETX']].apply(pd.to_numeric, errors='coerce')
    return df[df['S'] <= 220]

# Gather files
twiss_files = sorted(glob.glob("twiss_IQ875_*.out"))
groups = {
    "default": [],
    "10percentup": [],
    "10percentdown": [],
    "20percentup": [],
    "20percentdown": []
}

for file in twiss_files:
    base = os.path.basename(file).replace("twiss_IQ875_", "").replace(".out", "")
    hq_group = base.split('_')[0]
    if hq_group in groups:
        groups[hq_group].append((file, base))

# Assign styles and colormaps
styles = {
    "default": "-",
    "10percentup": "--",
    "10percentdown": ":",
    "20percentup": "-.",
    "20percentdown": (0, (3, 5, 1, 5))
}
colormaps = {
    "default": cm.get_cmap("tab10"),
    "10percentup": cm.get_cmap("Set2"),
    "10percentdown": cm.get_cmap("Dark2"),
    "20percentup": cm.get_cmap("cool"),
    "20percentdown": cm.get_cmap("autumn")
}

plt.figure(figsize=(14, 6))

for group, files in groups.items():
    cmap = colormaps[group]
    linestyle = styles[group]
    n = len(files)
    for i, (file, label) in enumerate(sorted(files)):
        df = read_twiss_file(file)
        betx_label = "_".join(label.split('_')[1:])
        color = to_hex(cmap(i / max(n - 1, 1)))
        plt.plot(df['S'], df['BETX'], label=f"{group} – {betx_label}", linestyle=linestyle, color=color, linewidth=1.5)

plt.title("BETX vs S for Varying HQ875 Currents and BETX")
plt.xlabel("S (m)")
plt.ylabel("BETX (m)")
plt.grid(True)

plt.legend(fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("betx_HQ875_readable.png", dpi=300)
plt.show()
