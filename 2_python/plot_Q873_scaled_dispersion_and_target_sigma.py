#!/usr/bin/env python3
"""
Scaled dispersion reconstruction and target sigma prediction vs Q873 current
Robust against filename variations (default / flipPol / legacy)
"""

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# =========================================================
# Paths
# =========================================================
#BASE = "."
#def P(p): return p if os.path.isabs(p) else os.path.join(BASE, p)
TWISS_DIR = "../1_twiss_outputs"

def P(fname):
    return fname if os.path.isabs(fname) else os.path.join(TWISS_DIR, fname)

# =========================================================
# Currents & labels
# =========================================================
order = ["-30","-24","-18","-12","-6","0","+6","+12","+18","+24","+30"]

xlabels = {
    "-30":"Q873 −30A","-24":"Q873 −24A","-18":"Q873 −18A","-12":"Q873 −12A","-6":"Q873 −6A",
    "0":"Q873 Baseline","+6":"Q873 +6A","+12":"Q873 +12A","+18":"Q873 +18A","+24":"Q873 +24A","+30":"Q873 +30A"
}

monitors = ["MW873","MW875","MW876"]

# =========================================================
# File maps
# =========================================================
files_flip = {
    "-30": "twiss_Q873_-30A_flipPol.tfs",
    "-24": "twiss_Q873_-24A_flipPol.tfs",
    "-18": "twiss_Q873_-18A_flipPol.tfs",
    "-12": "twiss_Q873_-12A_flipPol.tfs",
    "-6" : "twiss_Q873_-6A_flipPol.tfs",
    "+6" : "twiss_Q873_+6A_flipPol.tfs",
    "+12": "twiss_Q873_+12A_flipPol.tfs",
    "+18": "twiss_Q873_+18A_flipPol.tfs",
    "+24": "twiss_Q873_+24A_flipPol.tfs",
    "+30": "twiss_Q873_+30A_flipPol.tfs",
}

files_orig = {
    "-30": "twiss_Q873_minus30A.tfs",
    "-24": "twiss_Q873_minus24A.tfs",
    "-18": "twiss_Q873_minus18A.tfs",
    "-12": "twiss_Q873_minus12A.tfs",
    "-6" : "twiss_Q873_minus6A.tfs",
    "+6" : "twiss_Q873_plus6A.tfs",
    "+12": "twiss_Q873_plus12A.tfs",
    "+18": "twiss_Q873_plus18A.tfs",
    "+24": "twiss_Q873_plus24A.tfs",
    "+30": "twiss_Q873_plus30A.tfs",
}

# =========================================================
# Choose files (baseline explicitly handled)
# =========================================================
files = {}

# baseline — ALWAYS this
#baseline_file = P("twiss_Q873_default.tfs")
#if not os.path.exists(baseline_file):
 #   raise RuntimeError("Baseline file twiss_Q873_default.tfs not found")
#files["0"] = baseline_file

baseline_file = P("twiss_Q873_default.tfs")
if not os.path.exists(baseline_file):
    raise RuntimeError(f"Baseline file not found: {baseline_file}")
files["0"] = baseline_file

# others
for k in order:
    if k == "0":
        continue
    if k in files_flip and os.path.exists(P(files_flip[k])):
        files[k] = P(files_flip[k])
    elif k in files_orig and os.path.exists(P(files_orig[k])):
        files[k] = P(files_orig[k])

print("Using TFS files:")
for k in order:
    if k in files:
        print(f"  {k:>4} : {os.path.basename(files[k])}")

# =========================================================
# Measured σ table (mm)
# =========================================================
meas_data = {
    "Q873 − 30A":[2.30,5.05,3.56,4.50,2.29,4.40],
    "Q873 − 24A":[2.20,4.95,3.40,3.84,2.44,3.60],
    "Q873 − 18A":[2.29,5.01,3.81,3.20,2.78,2.89],
    "Q873 − 12A":[2.19,4.97,3.76,2.58,2.84,2.21],
    "Q873 − 6A" :[2.18,5.00,3.92,1.95,0.745,1.56],
    "Q873 Baseline":[2.293,4.959,4.957,1.371,3.771,0.9685],
    "Q873 + 6A":[2.22,5.01,3.92,4.22,3.60,0.538],
    "Q873 + 12A":[2.30,5.01,4.62,0.487,4.04,0.950],
    "Q873 + 18A":[2.32,5.03,4.81,0.918,4.32,1.64],
    "Q873 + 24A":[2.33,5.05,5.10,1.54,4.37,2.37],
    "Q873 + 30A":[2.30,5.02,5.14,2.23,4.67,3.07],
}

meas_df = pd.DataFrame.from_dict(
    meas_data, orient="index",
    columns=["873 H","873 V","875 H","875 V","876 H","876 V"]
)

def find_meas(label):
    key = label.replace(" ","")
    for idx in meas_df.index:
        if idx.replace(" ","") == key:
            return meas_df.loc[idx]
    raise KeyError(label)

# =========================================================
# TFS reader + helpers
# =========================================================
def read_tfs(path):
    with open(path,"r",errors="ignore") as f:
        lines=f.readlines()
    h=next(i for i,l in enumerate(lines) if l.lstrip().startswith("*"))
    cols=lines[h].replace("*","").split()
    start=h+1
    if start<len(lines) and lines[start].lstrip().startswith("$"):
        start+=1
    rows=[l for l in lines[start:] if l.strip() and not l.lstrip().startswith(("@","#","*","$"))]
    df=pd.read_csv(StringIO("".join(rows)),sep=r"\s+",names=cols,engine="python")
    df.columns=[c.upper() for c in df.columns]
    df["NAME"]=(df["NAME"].astype(str).str.upper()
                .str.replace(r"[:\.].*$","",regex=True))
    return df

def get_rows(df,names):
    out={}
    for n in names:
        m=df[df["NAME"]==n]
        if m.empty: m=df[df["NAME"].str.startswith(n)]
        if m.empty: m=df[df["NAME"].str.contains(n)]
        if m.empty:
            raise RuntimeError(f"{n} not found in TFS")
        out[n]=m.iloc[0]
    return out

def pick_target(df):
    for n in ["MTGT","MWTGT","TARGET","TGT"]:
        m=df[df["NAME"]==n]
        if not m.empty:
            return m.iloc[0]
    return df.loc[(df["S"]-207).abs().idxmin()]

# =========================================================
# Beam parameters
# =========================================================
p=8.83490; m0=0.9382720813
gamma=math.sqrt(1+(p/m0)**2)
beta=math.sqrt(1-1/gamma**2)
betagamma=beta*gamma

def eps95_to_eps(eps95):
    return eps95/6*1e-6/betagamma

# =========================================================
# Emittance (fallback constants)
# =========================================================
print("WARNING: using default emittances")
eps95x={xlabels[k]:20.0 for k in order}
eps95y={xlabels[k]:12.0 for k in order}
epsx={k:eps95_to_eps(eps95x[xlabels[k]]) for k in order}
epsy={k:eps95_to_eps(eps95y[xlabels[k]]) for k in order}

# =========================================================
# Fit σδ at baseline
# =========================================================
df0=read_tfs(files["0"])
rows0=get_rows(df0,monitors)
base=find_meas("Q873 Baseline")

Y=[]; X=[]
for w,c in zip(monitors,["873 H","875 H","876 H"]):
    r=rows0[w]
    sx=base[c]*1e-3
    Y.append(sx*sx-epsx["0"]*r["BETX"])
    X.append([r["DX"]**2])

#s2,_=np.linalg.lstsq(np.array(X),np.array(Y),rcond=None)
s2, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
sigma_delta=math.sqrt(max(s2[0],0.0))
print(f"βγ={betagamma:.6f}   fitted σδ={sigma_delta:.3e}")

# =========================================================
# Sweep currents
# =========================================================
Dx_est={w:[] for w in monitors}
Dx_mod={w:[] for w in monitors}
labels=[]
target_sig=[]

for k in order:
    if k not in files: continue
    df=read_tfs(files[k])
    rows=get_rows(df,monitors)
    meas=find_meas(xlabels[k])
    labels.append(xlabels[k])

    est=[]; mod=[]
    for w,c in zip(monitors,["873 H","875 H","876 H"]):
        r=rows[w]
        sx=meas[c]*1e-3
        bx=r["BETX"]; Dx=r["DX"]
        base=max(sx*sx-epsx[k]*bx,0.0)
        De=math.sqrt(base)/sigma_delta if sigma_delta>0 else 0.0
        De=math.copysign(De,Dx)
        Dx_est[w].append(De); Dx_mod[w].append(Dx)
        est.append(De); mod.append(Dx)

    kx=np.median([e/m for e,m in zip(est,mod) if abs(m)>1e-12])
    tgt=pick_target(df)
    sigx=math.sqrt(epsx[k]*tgt["BETX"]+(sigma_delta*kx*tgt["DX"])**2)*1e3
    sigy=math.sqrt(epsy[k]*tgt["BETY"]+(sigma_delta*kx*tgt["DY"])**2)*1e3
    target_sig.append((sigx,sigy))

# =========================================================
# Plots
# =========================================================
xi=np.arange(len(labels))

plt.figure(figsize=(11,6))
for w in monitors:
    plt.plot(xi,Dx_est[w],"-o",label=f"{w} Dx est")
    plt.plot(xi,Dx_mod[w],"--",label=f"{w} Dx model")
plt.xticks(xi,labels,rotation=30,ha="right")
plt.ylabel("D_x [m]")
plt.title("Estimated vs Model Dx")
plt.grid(True,ls=":")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(11,6))
plt.plot(xi,[s[0] for s in target_sig],"-o",label="σx @ target")
plt.plot(xi,[s[1] for s in target_sig],"-o",label="σy @ target")
plt.xticks(xi,labels,rotation=30,ha="right")
plt.ylabel("σ [mm]")
plt.title("Predicted σ at target (~207 m)")
plt.grid(True,ls=":")
plt.legend()
plt.tight_layout()
plt.show()

