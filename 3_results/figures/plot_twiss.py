import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def read_twiss_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Extract headers from the line starting with '*'
    header_line = next(line for line in lines if line.startswith('*')).strip().lstrip('*').split()

    # Read valid data lines (non-header, non-format)
    data_lines = [line.strip() for line in lines if not line.startswith(('*', '$')) and line.strip()]
    data = [line.split() for line in data_lines]

    # Clean and create DataFrame
    df = pd.DataFrame(data, columns=[col.strip('"') for col in header_line])

    # Ensure numeric types for key columns
    df = df[['S', 'BETX', 'BETY']].apply(pd.to_numeric, errors='coerce')

    # Filter S range to match MAD-X PDF plots
    df = df[df['S'] <= 220]

    return df

# Load all twiss_*.out files
twiss_files = sorted(glob.glob("twiss_*.out"))

# Plot BETX
plt.figure(figsize=(10, 6))
for file in twiss_files:
    df = read_twiss_file(file)
    label = os.path.basename(file).replace("twiss_", "").replace(".out", "")
    plt.plot(df['S'], df['BETX'], label=label)
plt.title("BETX vs S (0–220 m)")
plt.xlabel("S (m)")
plt.ylabel("BETX (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("betx_vs_s_final.png")
plt.show()

# Plot BETY
plt.figure(figsize=(10, 6))
for file in twiss_files:
    df = read_twiss_file(file)
    label = os.path.basename(file).replace("twiss_", "").replace(".out", "")
    plt.plot(df['S'], df['BETY'], label=label)
plt.title("BETY vs S (0–220 m)")
plt.xlabel("S (m)")
plt.ylabel("BETY (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bety_vs_s_final.png")
plt.show()
