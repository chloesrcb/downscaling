#%%
import os
import math
import numpy as np
import pandas as pd
import torch

from functions_dwscl import (
    EGPDPINN_NNOnly,
    train_egpd_nn_only,
    predict_params,
)

DATA_FOLDER = os.environ.get("DATA_FOLDER", "../phd_extremes/data/")
IM_FOLDER =os.environ.get("DATA_FOLDER", "../phd_extremes/thesis/resources/images/downscaling/")
DOWNSCALING_TABLE = os.path.join(DATA_FOLDER, "downscaling/downscaling_table.csv")
DOWNSCALING_TABLE = os.path.join(DATA_FOLDER, "downscaling/downscaling_table_named_2019_2025.csv")

OUT_COMPARISON = os.path.join(DATA_FOLDER, "downscaling/model_comparison.csv")

# %%
xi_init = 0.262
sigma_init = 0.591
kappa_init = 0.270

print("init:", {"sigma": sigma_init, "kappa": kappa_init, "xi": xi_init})

#%%
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";", header=0)
#%%
df_raw.columns
#%%
df_raw.head()
#%%
# check number of missing values in Y_obs
print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
# Prop of 0s in Y_obs
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).sum() / len(df_raw))

#%%
df_analysis = df_raw.copy()
# mean of X_is
# x_cols27 = [f"X{i}" for i in range(1, 28)]
x_cols27 = [col for col in df_analysis.columns if col.startswith("X")]
#%%
df_analysis["X_mean"] = df_analysis[x_cols27].mean(axis=1, skipna=True)
# summary of X_mean
print("X_mean summary:")
print(df_analysis["X_mean"].describe())

#%%

y_thr = [0., 0.22, 0.5, 1.0]
x_thr = [0., 0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

rows = []
for yt in y_thr:
    mask_y = df_analysis["Y_obs"] > yt
    for xt in x_thr:
        rain_x_any = (df_analysis.loc[mask_y, x_cols27] > xt).any(axis=1).mean()
        rows.append({"y_thr": yt, "x_thr": xt, "P(any X_i>x | Y>y)": rain_x_any})
pd.DataFrame(rows)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

y_thr = [0., 0.22, 0.5, 1.0]
x_thr = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

rows = []
for yt in y_thr:
    mask_y = df_analysis["Y_obs"] > yt
    for xt in x_thr:
        rain_x_any = (df_analysis.loc[mask_y, x_cols27] > xt).any(axis=1).mean()
        rows.append({"y_thr": yt, "x_thr": xt, "prob": rain_x_any})

#%%
df_plot = pd.DataFrame(rows)

plt.figure(figsize=(8,5))
sns.set_style("whitegrid") 

markers = ['s', '^', 'o', 'd'] 
palette = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7"
]
ax = sns.lineplot(
    data=df_plot,  
    x="x_thr", 
    y="prob", 
    hue="y_thr", 
    style="y_thr",
    markers=markers, 
    markersize=7,
    palette=palette,
    linewidth=2,
    dashes=False
)

plt.xlabel("$x$ threshold (mm/h)", fontsize=12)
plt.ylabel("$P(\max_i(X_i) > x \mid Y > y)$", fontsize=12)

plt.legend(
    title="$y$ threshold (mm/5 min)", 
    bbox_to_anchor=(1.05, 0.5), 
    loc='center left', 
    borderaxespad=0.,
    frameon=True
)

plt.xticks(x_thr, rotation=45)
plt.tight_layout()

# Save plot
os.makedirs(IM_FOLDER, exist_ok=True)
plt.savefig(os.path.join(IM_FOLDER, "conditional_prob_plot.png"), dpi=300, bbox_inches="tight")
print("Plot saved to:", os.path.join(IM_FOLDER, "conditional_prob_plot.png"))

plt.show()

#%%
# same but with P(Y>y | max(X)>x)
y_thr = [0., 0.22, 0.5, 1.0]
x_thr = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

rows = []
for xt in x_thr:
    mask_x = (df_analysis[x_cols27] > xt).any(axis=1)
    for yt in y_thr:
        rain_y_given_x = (df_analysis.loc[mask_x, "Y_obs"] > yt).mean()
        rows.append({"x_thr": xt, "y_thr": yt, "P(Y>y | max(X)>x)": rain_y_given_x})
df_plot_y_given_x = pd.DataFrame(rows)

#%%
plt.figure(figsize=(8,5))
sns.set_style("whitegrid")
markers = ['o']
palette = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7"
]
ax = sns.lineplot(
    data=df_plot_y_given_x,  
    x="y_thr", 
    y="P(Y>y | max(X)>x)", 
    hue="x_thr", 
    style="x_thr",
    markers=markers, 
    markersize=7,
    palette=palette,
    linewidth=2,
    dashes=False
)
plt.xlabel("$y$ threshold (mm/5 min)", fontsize=12)
plt.ylabel("$P(Y > y \mid \max_i(X_i) > x)$", fontsize=12)
plt.legend(
    title="$x$ threshold (mm/h)", 
    bbox_to_anchor=(1.05, 0.5), 
    loc='center left', 
    borderaxespad=0.,
    frameon=True
)
plt.xticks(y_thr, rotation=45)
plt.tight_layout()
os.makedirs(IM_FOLDER, exist_ok=True)
plt.savefig(os.path.join(IM_FOLDER, "conditional_prob_y_given_x_plot.png"), dpi=300, bbox_inches="tight")
print("Plot saved to:", os.path.join(IM_FOLDER, "conditional_prob_y_given_x_plot.png"))
plt.show()

#%%
x_cols27 = [col for col in df_analysis.columns if col.startswith("X") and col != "X_mean"]

df_analysis["X_mean"] = df_analysis[x_cols27].mean(axis=1, skipna=True)
df_analysis["X_max"]  = df_analysis[x_cols27].max(axis=1, skipna=True)
df_analysis["X_min"]  = df_analysis[x_cols27].min(axis=1, skipna=True)
df_analysis["X_sd"]   = df_analysis[x_cols27].std(axis=1, skipna=True, ddof=1)
df_analysis["X_q90"]  = df_analysis[x_cols27].quantile(0.90, axis=1)
df_analysis["X_q95"]  = df_analysis[x_cols27].quantile(0.95, axis=1)

#%%
import matplotlib.pyplot as plt
import numpy as np

vars_compare = ["Y_obs", "X_mean", "X_max", "X_q90"]

plt.figure(figsize=(8, 5))
for v in vars_compare:
    arr = df_analysis[v].dropna().to_numpy()
    arr = arr[arr >= 0]
    plt.hist(arr, bins=80, density=True, histtype="step", linewidth=2, label=v)

plt.xlim(0, np.quantile(df_analysis["X_max"].dropna(), 0.99))
plt.xlabel("Rainfall")
plt.ylabel("Density")
plt.title("Marginal distributions")
plt.legend()
plt.grid(True)
plt.show()

#%%

rain_y = df_analysis["Y_obs"] > 0
rain_x = (df_analysis[x_cols27] > 0).any(axis=1)

TP = (rain_x & rain_y).sum()
FP = (rain_x & ~rain_y).sum()
FN = (~rain_x & rain_y).sum()
TN = (~rain_x & ~rain_y).sum()

print("P(Y) =", rain_y.mean())
print("P(X) =", rain_x.mean())
print("P(Y|X) =", TP / (TP + FP) if (TP+FP)>0 else np.nan)
print("P(X|Y) =", TP / (TP + FN) if (TP+FN)>0 else np.nan)
print("FPR =", FP / (FP + TN) if (FP+TN)>0 else np.nan)
print("FNR =", FN / (FN + TP) if (FN+TP)>0 else np.nan)

#%%
import numpy as np
import matplotlib.pyplot as plt

rain_y = df_analysis["Y_obs"] > 0
rain_x = (df_analysis[x_cols27] > 0).any(axis=1)

TP = (rain_x & rain_y).sum()
FP = (rain_x & ~rain_y).sum()
FN = (~rain_x & rain_y).sum()
TN = (~rain_x & ~rain_y).sum()

cm = np.array([[TP, FP],
               [FN, TN]])

labels = np.array([["TP", "FP"],
                   ["FN", "TN"]])

pY   = rain_y.mean()
pX   = rain_x.mean()
pY_X = TP / (TP + FP) if (TP+FP)>0 else np.nan   # precision = P(Y|X)
pX_Y = TP / (TP + FN) if (TP+FN)>0 else np.nan   # recall    = P(X|Y)
fpr  = FP / (FP + TN) if (FP+TN)>0 else np.nan
fnr  = FN / (FN + TP) if (FN+TP)>0 else np.nan

#%%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6.2, 5.4))
cmap = plt.cm.Reds 
im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

ax.set_xticks([0, 1], labels=["Observed rain\n(Y > 0)", "No rain\n(Y = 0)"])
ax.set_yticks([0, 1], labels=["Detected\n(max($X_i$) > 0)", "Not detected\n(max($X_i$) =0)"])

ax.grid(False)

total = cm.sum()
color_threshold = cm.max() / 2.
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = (val / total * 100) if total > 0 else 0.0
        text_color = "white" if val > color_threshold else "black"

        ax.text(
            j, i,
            f"{labels[i, j]}\n{pct:.2f}%",
            ha="center", va="center",
            color=text_color,
            fontsize=10,
            fontweight="bold"
        )

metrics_txt = (
    f"$P(Y>0) = {pY:.3f}$\n"
    f"$P(\max(X_i)>0) = {pX:.3f}$\n"
    f"$P(Y>0|\max(X_i)>0) = {pY_X:.3f}$\n"
    f"$P(\max(X_i)>0|Y>0) = {pX_Y:.3f}$\n"
    f"FPR = {fpr:.3f}\nFNR = {fnr:.3f}"
)

ax.text(1.05, 0.5, metrics_txt,
        transform=ax.transAxes,
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8),
        fontsize=10)


plt.tight_layout()

os.makedirs(IM_FOLDER, exist_ok=True)
plt.savefig(os.path.join(IM_FOLDER, "event_detection_cm.png"), dpi=300, bbox_inches="tight")
plt.show()

#%%
d = df_raw.copy()
d["time"] = pd.to_datetime(d["time"], utc=True)

d["hour_id"] = d["time"].dt.floor("h")

Y1h = d.groupby(["station","hour_id"])["Y_obs"].sum().rename("Y_sum_1h")
Ymax = d.groupby(["station","hour_id"])["Y_obs"].max().rename("Y_max_1h")
Yany = (d["Y_obs"]>0).groupby([d["station"], d["hour_id"]]).any().astype(int).rename("Y_any_1h")

d = d.join(Y1h, on=["station","hour_id"]).join(Ymax, on=["station","hour_id"]).join(Yany, on=["station","hour_id"])

#%%
# X_mean
d["X_mean"] = d[x_cols27].mean(axis=1, skipna=True)
d["X_max"] = d[x_cols27].max(axis=1, skipna=True)
d["X_min"] = d[x_cols27].min(axis=1, skipna=True)
d["X_sd"] = d[x_cols27].std(axis=1, skipna=True, ddof=1)
#%%
# crosstabs
lon1 = np.deg2rad(d["lon_X"].to_numpy(dtype=float))
lat1 = np.deg2rad(d["lat_X"].to_numpy(dtype=float))
lon2 = np.deg2rad(d["lon_Y"].to_numpy(dtype=float))
lat2 = np.deg2rad(d["lat_Y"].to_numpy(dtype=float))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
d["spatial_dist_XY_km"] = 6371.0 * c

dist_bins = pd.qcut(d["spatial_dist_XY_km"], 1)

y_mask = d["Y_sum_1h"] > 1
x_mask = (d[x_cols27] > 2).any(axis=1)

out = d.groupby(dist_bins).apply(lambda d: pd.Series({
    "n": len(d),
    "P(Y|X)": ((d["Y_sum_1h"] > 1) & ((d[x_cols27] > 2).any(axis=1))).sum()
             / (((d[x_cols27] > 1).any(axis=1)).sum() + 1e-12),
    "P(X|Y)": (((d[x_cols27] > 1).any(axis=1)) & (d["Y_sum_1h"] > 1)).sum()
             / (((d["Y_sum_1h"] > 1)).sum() + 1e-12),
    "P(X)": ((d[x_cols27] > 1).any(axis=1)).mean(),
    "P(Y)": (d["Y_sum_1h"] > 1).mean(),
}))
print(out)

#%%
# correlation X_mean and Y_sum_1h
print("Correlation X_mean and Y_sum_1h:", d[["X_mean", "Y_sum_1h"]].corr(method = "pearson").iloc[0,1])
print("Correlation X_mean and Y_obs:", d[["X_mean", "Y_obs"]].corr(method = "pearson").iloc[0,1])

# %%
