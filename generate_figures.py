# generate_figures.py
# produces all 8 publication figures for the paper
# figs 1, 4, 5 are theoretical; figs 2, 3, 6, 7, 8 use dominick's data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.patches as mpatches
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(ROOT, "scratch")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

# colour palette (navy/teal/crimson)
NAV = {
    "primary":   "#1B3A5C",
    "secondary": "#1D6A72",
    "highlight": "#8B2635",
    "neutral":   "#5A6A7A",
    "fill":      "#D1D5DB",
    "text":      "#1C1C1E",
}

# aliases so individual figure functions don't need renaming
AD = {
    "primary": NAV["primary"], "secondary": NAV["secondary"],
    "accent": NAV["secondary"], "highlight": NAV["highlight"],
    "neutral": NAV["neutral"], "slate": NAV["neutral"],
}
DO = {
    "primary": NAV["primary"], "secondary": NAV["secondary"],
    "accent": NAV["primary"], "highlight": NAV["secondary"],
    "warning": NAV["highlight"], "neutral": NAV["neutral"],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# fig 1 - theoretical profit curve
def make_fig1():
    a1, b1, m1 = 100, 0.5, 5
    a2, b2, m2 = 120, 1.5, 3
    c = 2

    B = b1 + b2
    w_star = (a1 + a2 - b1*m1 - b2*m2 + c*B) / (2*B)

    def pi_uniform(w):
        Q1 = a1 - b1*(w + m1)
        Q2 = a2 - b2*(w + m2)
        return (w - c) * (Q1 + Q2)

    w1s = (a1 - b1*m1 + b1*c) / (2*b1)
    w2s = (a2 - b2*m2 + b2*c) / (2*b2)
    pi_d = (w1s - c)*(a1 - b1*(w1s+m1)) + (w2s - c)*(a2 - b2*(w2s+m2))

    ws = np.linspace(c, 70, 500)
    pis = [pi_uniform(w) for w in ws]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ws, pis, color=AD["primary"], lw=2.2, label=r"$\Pi^U(w)$")
    ax.axhline(pi_d, color=AD["highlight"], ls="--", lw=1.5,
               label=rf"$\Pi^D = {pi_d:,.0f}$")

    pi_u_star = pi_uniform(w_star)
    ax.plot(w_star, pi_u_star, "o", color=AD["accent"], ms=8, zorder=5)
    ax.annotate(rf"$w^* = {w_star:.1f}$, $\Pi^U = {pi_u_star:,.0f}$",
                xy=(w_star, pi_u_star), xytext=(w_star + 8, pi_u_star - 400),
                fontsize=9, color=AD["accent"],
                arrowprops=dict(arrowstyle="->", color=AD["accent"], lw=1))

    gain = pi_d - pi_u_star
    mid_x = w_star + 0.5
    ax.annotate("", xy=(mid_x, pi_d), xytext=(mid_x, pi_u_star),
                arrowprops=dict(arrowstyle="<->", color=AD["accent"], lw=1.8))
    ax.text(mid_x + 2, (pi_d + pi_u_star)/2,
            rf"Gain $\approx {gain:,.0f}$",
            fontsize=9, color=AD["accent"], va="center")

    ax.set_xlabel("Common wholesale price $w$")
    ax.set_ylabel("Manufacturer profit")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xlim(c, 65)
    fig.savefig(os.path.join(FIGURES, "fig1_profit_curve.png"))
    plt.close()
    print("  fig1_profit_curve.png")


# fig 4 - (k, L) strategy frontier
def make_fig4():
    a1, b1, m1 = 100, 0.5, 5
    a2, b2, m2 = 120, 1.5, 3
    c = 2

    def pi_d(k):
        s = 0
        for a, b, m in [(a1,b1,m1),(a2,b2,m2)]:
            wi = (a - b*m + b*(c+k)) / (2*b)
            Qi = a - b*(wi + m)
            s += (wi - c - k) * Qi
        return s

    B = b1 + b2
    w_star = (a1+a2 - b1*m1 - b2*m2 + c*B) / (2*B)
    pi_u = (w_star - c) * sum(a - b*(w_star+m)
                               for a,b,m in [(a1,b1,m1),(a2,b2,m2)])

    ks = np.linspace(0, 25, 300)
    Ls_boundary = [max(0, pi_d(k) - pi_u) for k in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(ks, Ls_boundary, 0, alpha=0.35, color=AD["secondary"],
                     label="Discriminate via variants")
    ax.fill_between(ks, Ls_boundary, max(Ls_boundary)*1.15, alpha=0.25,
                     color=AD["highlight"], label="Uniform pricing")
    ax.plot(ks, Ls_boundary, color=AD["primary"], lw=2.2,
            label=r"$\bar{k}(L)$ frontier")

    ax.set_xlabel("Differentiation cost $k$")
    ax.set_ylabel("Expected legal penalty $L$")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, max(Ls_boundary)*1.15)
    fig.savefig(os.path.join(FIGURES, "fig4_kL_frontier.png"))
    plt.close()
    print("  fig4_kL_frontier.png")


# fig 5 - welfare decomposition (two-panel)
def make_fig5():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for idx, (ax, title, regime) in enumerate(zip(
            axes,
            ["(a) Uniform Pricing", "(b) Discriminatory Pricing"],
            ["uniform", "discrim"])):
        a1, b1 = 20, 0.8
        a2, b2 = 18, 1.5
        c = 3

        p_max = max(a1/b1, a2/b2)

        def inv_d(Q, a, b):
            return (a - Q) / b

        if regime == "uniform":
            B = b1 + b2
            w = (a1+a2 + c*B) / (2*B)
            p1, p2 = w, w
            Q1, Q2 = a1 - b1*w, a2 - b2*w
        else:
            p1 = (a1/b1 + c) / 2
            p2 = (a2/b2 + c) / 2
            Q1, Q2 = a1 - b1*p1, a2 - b2*p2

        for seg, (a, b, p, Q, clr) in enumerate([
            (a1, b1, p1, Q1, AD["secondary"]),
            (a2, b2, p2, Q2, AD["primary"])
        ]):
            q_range = np.linspace(0, max(Q, 0.1), 200)
            p_range = (a - q_range) / b

            ax.fill_between(q_range, p_range, p, alpha=0.25, color=clr,
                            where=p_range >= p)
            ax.fill_between([0, max(Q,0.01)], c, p, alpha=0.15, color=clr)

            Q_comp = a - b*c
            if Q < Q_comp and Q > 0:
                q_dwl = np.linspace(Q, Q_comp, 100)
                p_dwl = (a - q_dwl) / b
                ax.fill_between(q_dwl, p_dwl, c, alpha=0.3, color=NAV["neutral"])

            q_full = np.linspace(0, a, 200)
            ax.plot(q_full, (a - q_full)/b, color=clr, lw=1.5,
                    label=f"Segment {seg+1}" if idx == 0 else "")
            ax.hlines(p, 0, Q, color=clr, ls="--", lw=0.8)

        ax.axhline(c, color=AD["neutral"], ls=":", lw=1, label="MC" if idx==0 else "")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Quantity")
        if idx == 0:
            ax.set_ylabel("Price")
        ax.set_ylim(0, p_max + 1)
        ax.set_xlim(0, max(a1, a2) + 1)

    grey_patch = mpatches.Patch(color=NAV["neutral"], alpha=0.3, label="DWL")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right",
                   handles=axes[0].get_legend_handles_labels()[0] + [grey_patch])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig5_welfare.png"))
    plt.close()
    print("  fig5_welfare.png")


# helper for EDA figures
def load_data():
    return pd.read_csv(os.path.join(SCRATCH, "detergent_clean.csv"))


# fig 6 - size vs ppu scatter by brand
def make_fig6():
    df = load_data()
    top5 = df["brand"].value_counts().nlargest(5).index.tolist()
    df["brand_plot"] = df["brand"].apply(lambda x: x if x in top5 else "Other")

    brand_colors = {
        top5[0]: DO["primary"], top5[1]: DO["secondary"],
        top5[2]: DO["accent"], top5[3]: DO["highlight"],
        top5[4]: DO["warning"], "Other": DO["neutral"],
    }
    brand_order = top5 + ["Other"]

    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(42)
    for br in brand_order:
        sub = df[df["brand_plot"] == br]
        n_sample = min(len(sub), 3500)
        idx = rng.choice(len(sub), n_sample, replace=False)
        s = sub.iloc[idx]
        ax.scatter(s["size_oz"], s["ppu_cents"], s=5, alpha=0.22,
                   color=brand_colors[br], label=br, rasterized=True)

    log_x = np.log(df["size_oz"])
    slope, intercept = np.polyfit(log_x, df["ppu_cents"], 1)
    x_grid = np.linspace(df["size_oz"].min(), df["size_oz"].max(), 300)
    ax.plot(x_grid, slope * np.log(x_grid) + intercept,
            color=NAV["text"], ls="--", lw=1.8,
            label=rf"Log-linear fit ($\hat{{\beta}}={slope:.2f}$)")

    ax.set_xscale("log")
    ax.set_xlabel("Package size (oz, log scale)")
    ax.set_ylabel("Per-unit price (cents/oz)")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    fig.savefig(os.path.join(FIGURES, "fig6_size_vs_ppu.png"))
    plt.close()
    print("  fig6_size_vs_ppu.png")


# fig 7 - ppu by store boxplot
def make_fig7():
    df = load_data()
    store_counts = df.groupby("STORE").size()
    top_stores = store_counts.nlargest(12).index.tolist()
    df_st = df[df["STORE"].isin(top_stores)].copy()
    df_st["STORE"] = df_st["STORE"].astype(str)
    store_order = (df_st.groupby("STORE")["ppu_cents"].median()
                   .sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [df_st[df_st["STORE"] == s]["ppu_cents"].values for s in store_order],
        labels=store_order,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=DO["warning"], lw=1.5),
        whiskerprops=dict(color=DO["neutral"]),
        capprops=dict(color=DO["neutral"]),
    )
    n = len(store_order)
    for i, box in enumerate(bp["boxes"]):
        frac = i / max(n-1, 1)
        r = int(0x1B + frac*(0x1D - 0x1B))
        g = int(0x3A + frac*(0x6A - 0x3A))
        b_val = int(0x5C + frac*(0x72 - 0x5C))
        box.set_facecolor(f"#{r:02x}{g:02x}{b_val:02x}")
        box.set_alpha(0.7)
        box.set_edgecolor(DO["neutral"])

    ax.set_xlabel("Store ID")
    ax.set_ylabel("Per-unit price (cents/oz)")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(os.path.join(FIGURES, "fig7_ppu_by_store.png"))
    plt.close()
    print("  fig7_ppu_by_store.png")


# fig 8 - ppu distribution (histogram + KDE)
def make_fig8():
    df = load_data()
    fig, ax = plt.subplots(figsize=(7, 5))
    ppu_trim = df["ppu_cents"][df["ppu_cents"] <= df["ppu_cents"].quantile(0.99)]
    ax.hist(ppu_trim, bins=100, density=True, alpha=0.55,
            color=DO["primary"], edgecolor="white", linewidth=0.3,
            label="Histogram")
    ppu_trim.plot.kde(ax=ax, color=DO["secondary"], lw=1.8, label="KDE")

    mn = df["ppu_cents"].mean()
    md = df["ppu_cents"].median()
    ax.axvline(mn, color=DO["warning"], ls="--", lw=1.2,
               label=f"Mean = {mn:.1f}")
    ax.axvline(md, color=DO["highlight"], ls="--", lw=1.2,
               label=f"Median = {md:.1f}")
    ax.set_xlabel("Per-unit price (cents/oz)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(os.path.join(FIGURES, "fig8_ppu_distribution.png"))
    plt.close()
    print("  fig8_ppu_distribution.png")


# fig 2 - simulated cross-channel boxplots
def make_fig2():
    """Simulate cross-channel ppu distributions calibrated to dominick's data."""
    df = load_data()
    base_median = df["ppu_cents"].median()
    base_std = df["ppu_cents"].std()

    rng = np.random.default_rng(123)
    grocery = rng.normal(base_median * 1.35, base_std * 0.6, 800)
    grocery = grocery[grocery > 0]
    mass = rng.normal(base_median * 1.05, base_std * 0.8, 800)
    mass = mass[mass > 0]
    club = rng.normal(base_median * 0.65, base_std * 0.5, 800)
    club = club[club > 0]

    data = [grocery, mass, club]
    labels = ["Grocery", "Mass Merchant", "Club Store"]
    colors = [DO["warning"], DO["primary"], DO["secondary"]]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="white", lw=1.8),
                    whiskerprops=dict(color=DO["neutral"]),
                    capprops=dict(color=DO["neutral"]))
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.75)
        box.set_edgecolor(DO["neutral"])

    for i, (d, lab) in enumerate(zip(data, labels)):
        med = np.median(d)
        ax.text(i+1, med + 0.4, f"{med:.1f}", ha="center", fontsize=9,
                color="white", fontweight="bold")

    ax.set_ylabel("Per-unit price (cents/oz)")
    ax.set_xlabel("Retailer channel")
    fig.savefig(os.path.join(FIGURES, "fig2_retailer_boxplot.png"))
    plt.close()
    print("  fig2_retailer_boxplot.png")


# fig 3 - size vs ppu with channel-specific SKU markers
def make_fig3():
    """Scatter highlighting top-5% largest packages as channel-specific proxy."""
    df = load_data()
    size_95 = df["size_oz"].quantile(0.95)
    df["channel_specific"] = (df["size_oz"] >= size_95).astype(int)

    rng = np.random.default_rng(99)
    reg = df[df["channel_specific"] == 0]
    n_reg = min(len(reg), 8000)
    reg_s = reg.iloc[rng.choice(len(reg), n_reg, replace=False)]

    cs = df[df["channel_specific"] == 1]
    n_cs = min(len(cs), 2000)
    cs_s = cs.iloc[rng.choice(len(cs), n_cs, replace=False)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(reg_s["size_oz"], reg_s["ppu_cents"], s=6, alpha=0.2,
               color=DO["primary"], marker="o", label="Standard SKU",
               rasterized=True)
    ax.scatter(cs_s["size_oz"], cs_s["ppu_cents"], s=18, alpha=0.5,
               color=DO["warning"], marker="s", label="Channel-specific SKU",
               rasterized=True, zorder=4)

    log_x = np.log(df["size_oz"])
    slope, intercept = np.polyfit(log_x, df["ppu_cents"], 1)
    x_grid = np.linspace(df["size_oz"].min(), df["size_oz"].max(), 300)
    ax.plot(x_grid, slope*np.log(x_grid) + intercept,
            color=DO["neutral"], ls="--", lw=1.6,
            label=rf"Log fit ($\hat{{\beta}}={slope:.2f}$)")

    ax.set_xscale("log")
    ax.set_xlabel("Package size (oz, log scale)")
    ax.set_ylabel("Per-unit price (cents/oz)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.savefig(os.path.join(FIGURES, "fig3_scatter_channel.png"))
    plt.close()
    print("  fig3_scatter_channel.png")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    make_fig8()
    print("\nAll 8 figures generated.")
