# eda_and_figures.py
# generates summary statistics table and three initial EDA figures
# (these are superseded by generate_figures.py for the final paper,
# but kept for reproducibility of the early exploratory analysis)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(ROOT, "scratch")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

df = pd.read_csv(os.path.join(SCRATCH, "detergent_clean.csv"))
print(f"Loaded {len(df):,} rows")

# summary statistics table
desc = df[["PRICE", "size_oz", "ppu_cents"]].describe().T
desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

display_names = {
    "PRICE": "Price (\\$)",
    "size_oz": "Size (oz)",
    "ppu_cents": "Per-unit price (\\textcent/oz)",
}

with open(os.path.join(ROOT, "summary_stats.tex"), "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Summary Statistics: Dominick's Laundry Detergent Scanner Data}\n")
    f.write("\\label{tab:summary}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l rrrr rrrr}\n\\toprule\n")
    f.write(" & $N$ & Mean & SD & Min & P25 & Median & P75 & Max \\\\\n")
    f.write("\\midrule\n")
    for orig, nice in display_names.items():
        row = desc.loc[orig]
        n = f"{row['count']:,.0f}"
        rest = " & ".join(f"{row[c]:.2f}" for c in ["mean","std","min","25%","50%","75%","max"])
        f.write(f"{nice} & {n} & {rest} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\n")
    f.write("\\medskip\\par\\noindent\\footnotesize\n")
    f.write(f"\\textit{{Notes:}} Dominick's Finer Foods scanner data, laundry detergent category. "
            f"Sample restricted to store-week-UPC observations with positive price, positive unit "
            f"movement, and ounce-convertible package sizes. "
            f"$N = {len(df):,}$ observations across "
            f"{df['STORE'].nunique()} stores, {df['UPC'].nunique()} UPCs, "
            f"and {df['WEEK'].nunique()} weeks.\n")
    f.write("\\end{table}\n")
print("Wrote summary_stats.tex")

desc.to_csv(os.path.join(SCRATCH, "summary_stats.csv"))

# fig 1: size vs ppu scatter
top5 = df["brand"].value_counts().nlargest(5).index.tolist()
df["brand_plot"] = df["brand"].apply(lambda x: x if x in top5 else "Other")
brand_order = top5 + ["Other"]
palette = dict(zip(brand_order, sns.color_palette("Set2", len(brand_order))))

fig, ax = plt.subplots(figsize=(7, 5))
rng = np.random.default_rng(42)
for br in brand_order:
    sub = df[df["brand_plot"] == br]
    n_sample = min(len(sub), 4000)
    idx = rng.choice(len(sub), n_sample, replace=False)
    s = sub.iloc[idx]
    ax.scatter(s["size_oz"], s["ppu_cents"], s=6, alpha=0.25,
               color=palette[br], label=br, rasterized=True)

log_x = np.log(df["size_oz"])
slope, intercept = np.polyfit(log_x, df["ppu_cents"], 1)
x_grid = np.linspace(df["size_oz"].min(), df["size_oz"].max(), 300)
ax.plot(x_grid, slope * np.log(x_grid) + intercept,
        "k--", lw=1.5, label=f"Log-linear fit ($\\hat\\beta$={slope:.2f})")

ax.set_xscale("log")
ax.set_xlabel("Package size (oz, log scale)")
ax.set_ylabel("Per-unit price (cents/oz)")
ax.set_title("Figure 1: Package Size and Per-Unit Price")
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(ROOT, "fig1_size_vs_ppu.png"))
plt.close()
print("Wrote fig1_size_vs_ppu.png")

# fig 2: ppu by store boxplot
store_counts = df.groupby("STORE").size()
top_stores = store_counts.nlargest(12).index.tolist()
df_st = df[df["STORE"].isin(top_stores)].copy()
df_st["STORE"] = df_st["STORE"].astype(str)
store_order = (df_st.groupby("STORE")["ppu_cents"].median()
               .sort_values().index.tolist())

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_st, x="STORE", y="ppu_cents", order=store_order,
            fliersize=0.8, linewidth=0.7, palette="Blues", ax=ax,
            showfliers=False)
ax.set_xlabel("Store ID")
ax.set_ylabel("Per-unit price (cents/oz)")
ax.set_title("Figure 2: Per-Unit Price Distribution by Store")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=45)
fig.savefig(os.path.join(ROOT, "fig2_ppu_by_store.png"))
plt.close()
print("Wrote fig2_ppu_by_store.png")

# fig 3: ppu distribution
fig, ax = plt.subplots(figsize=(7, 5))
ppu_trim = df["ppu_cents"][df["ppu_cents"] <= df["ppu_cents"].quantile(0.99)]
ax.hist(ppu_trim, bins=100, density=True, alpha=0.55, color="steelblue",
        edgecolor="white", linewidth=0.3, label="Histogram")
ppu_trim.plot.kde(ax=ax, color="darkblue", lw=1.5, label="KDE")
mn = df["ppu_cents"].mean()
md = df["ppu_cents"].median()
ax.axvline(mn, color="red", ls="--", lw=1, label=f"Mean = {mn:.1f}")
ax.axvline(md, color="orange", ls="--", lw=1, label=f"Median = {md:.1f}")
ax.set_xlabel("Per-unit price (cents/oz)")
ax.set_ylabel("Density")
ax.set_title("Figure 3: Distribution of Per-Unit Prices")
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(ROOT, "fig3_ppu_distribution.png"))
plt.close()
print("Wrote fig3_ppu_distribution.png")

print("\nAll figures and summary table generated.")
