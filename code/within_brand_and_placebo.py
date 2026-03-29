# within_brand_and_placebo.py
# Item 14: within-brand regressions (Tide, All, Purex)
# Item 15: placebo check — total shelf price as DV

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "scratch")
TEX = os.path.join(ROOT, "tex")
os.makedirs(TEX, exist_ok=True)

df = pd.read_csv(os.path.join(SCRATCH, "detergent_clean.csv"))
print(f"Loaded {len(df):,} rows")

df["STORE"] = df["STORE"].astype("category")
df["brand"] = df["brand"].astype("category")


def star(p):
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


# ── ITEM 14: Within-brand regressions ──────────────────────────────────────
brands = ["Tide", "All", "Purex"]
brand_results = {}

for br in brands:
    sub = df[df["brand"] == br].copy()
    n = len(sub)
    n_upc = sub["UPC"].nunique()
    print(f"\n{br}: {n:,} obs, {n_upc} UPCs")

    m = smf.ols("ppu_cents ~ log_size + C(STORE) + sale_flag", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["UPC"]})

    brand_results[br] = {
        "model": m,
        "n": n,
        "n_upc": n_upc,
        "beta": m.params["log_size"],
        "se": m.bse["log_size"],
        "p": m.pvalues["log_size"],
        "sale_coef": m.params["sale_flag"],
        "sale_se": m.bse["sale_flag"],
        "sale_p": m.pvalues["sale_flag"],
        "r2": m.rsquared,
        "r2_adj": m.rsquared_adj,
    }
    print(f"  log_size = {m.params['log_size']:.4f} "
          f"(SE={m.bse['log_size']:.4f}, p={m.pvalues['log_size']:.6f})")
    print(f"  sale_flag = {m.params['sale_flag']:.4f}")
    print(f"  R2 = {m.rsquared:.4f}")

# write within-brand table
with open(os.path.join(TEX, "within_brand_table.tex"), "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Within-Brand Size--Price Gradient}\n")
    f.write("\\label{tab:within_brand}\n\\small\n")
    f.write("\\begin{tabular}{l" + "c" * len(brands) + "}\n\\toprule\n")
    f.write(" & " + " & ".join(brands) + " \\\\\n")
    f.write("\\midrule\n")

    # log_size row
    coefs = []
    ses = []
    for br in brands:
        r = brand_results[br]
        coefs.append(f"${r['beta']:.3f}{star(r['p'])}$")
        ses.append(f"$({r['se']:.3f})$")
    f.write("$\\ln(\\text{Size})$ & " + " & ".join(coefs) + " \\\\\n")
    f.write(" & " + " & ".join(ses) + " \\\\\n[4pt]\n")

    # sale_flag row
    coefs = []
    ses = []
    for br in brands:
        r = brand_results[br]
        coefs.append(f"${r['sale_coef']:.3f}{star(r['sale_p'])}$")
        ses.append(f"$({r['sale_se']:.3f})$")
    f.write("Sale indicator & " + " & ".join(coefs) + " \\\\\n")
    f.write(" & " + " & ".join(ses) + " \\\\\n")
    f.write("\\midrule\n")

    f.write("Store FE & " + " & ".join(["Yes"] * len(brands)) + " \\\\\n")
    f.write("\\midrule\n")

    f.write("$R^2$ & " + " & ".join(
        f"{brand_results[br]['r2']:.3f}" for br in brands) + " \\\\\n")
    f.write("Adj.\\ $R^2$ & " + " & ".join(
        f"{brand_results[br]['r2_adj']:.3f}" for br in brands) + " \\\\\n")
    f.write("$N$ & " + " & ".join(
        f"{brand_results[br]['n']:,}" for br in brands) + " \\\\\n")
    f.write("UPCs & " + " & ".join(
        f"{brand_results[br]['n_upc']}" for br in brands) + " \\\\\n")

    f.write("\\bottomrule\n\\end{tabular}\n\n")
    f.write("\\medskip\\par\\noindent\\footnotesize\n")
    f.write("\\textit{Notes:} Dependent variable is per-unit price in cents per ounce. "
            "Each column restricts the sample to a single brand. "
            "Standard errors clustered by UPC in parentheses. "
            "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$. "
            "All specifications include store fixed effects and a sale indicator.\n")
    f.write("\\end{table}\n")

print("\nWrote tex/within_brand_table.tex")


# ── ITEM 15: Placebo check ────────────────────────────────────────────────
# DV1: per-unit price (ppu_cents) — should fall with size
# DV2: total shelf price (unit_price) — should rise with size
print("\n\n=== PLACEBO CHECK ===")

m_ppu = smf.ols("ppu_cents ~ log_size + C(STORE) + C(brand) + sale_flag",
                 data=df).fit(cov_type="cluster", cov_kwds={"groups": df["UPC"]})
m_shelf = smf.ols("unit_price ~ log_size + C(STORE) + C(brand) + sale_flag",
                   data=df).fit(cov_type="cluster", cov_kwds={"groups": df["UPC"]})

print(f"Per-unit price:   log_size = {m_ppu.params['log_size']:.4f} "
      f"(SE={m_ppu.bse['log_size']:.4f}), R2={m_ppu.rsquared:.4f}")
print(f"Total shelf price: log_size = {m_shelf.params['log_size']:.4f} "
      f"(SE={m_shelf.bse['log_size']:.4f}), R2={m_shelf.rsquared:.4f}")

placebo_models = [m_ppu, m_shelf]
placebo_labels = ["Per-unit price", "Shelf price"]
col_labels = ["(1) PPU (\\textcent/oz)", "(2) Shelf price (\\$)"]

with open(os.path.join(TEX, "placebo_table.tex"), "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Placebo Check: Per-Unit Price vs.\\ Total Shelf Price}\n")
    f.write("\\label{tab:placebo}\n\\small\n")
    f.write("\\begin{tabular}{lcc}\n\\toprule\n")
    f.write(" & " + " & ".join(col_labels) + " \\\\\n")
    f.write("\\midrule\n")

    # log_size
    coefs, ses = [], []
    for m in placebo_models:
        c, s, p = m.params["log_size"], m.bse["log_size"], m.pvalues["log_size"]
        coefs.append(f"${c:.3f}{star(p)}$")
        ses.append(f"$({s:.3f})$")
    f.write("$\\ln(\\text{Size})$ & " + " & ".join(coefs) + " \\\\\n")
    f.write(" & " + " & ".join(ses) + " \\\\\n[4pt]\n")

    # sale_flag
    coefs, ses = [], []
    for m in placebo_models:
        c, s, p = m.params["sale_flag"], m.bse["sale_flag"], m.pvalues["sale_flag"]
        coefs.append(f"${c:.3f}{star(p)}$")
        ses.append(f"$({s:.3f})$")
    f.write("Sale indicator & " + " & ".join(coefs) + " \\\\\n")
    f.write(" & " + " & ".join(ses) + " \\\\\n")
    f.write("\\midrule\n")

    f.write("Store FE & Yes & Yes \\\\\n")
    f.write("Brand FE & Yes & Yes \\\\\n")
    f.write("\\midrule\n")

    f.write("$R^2$ & " + " & ".join(
        f"{m.rsquared:.3f}" for m in placebo_models) + " \\\\\n")
    f.write("Adj.\\ $R^2$ & " + " & ".join(
        f"{m.rsquared_adj:.3f}" for m in placebo_models) + " \\\\\n")
    f.write("$N$ & " + " & ".join(
        f"{int(m.nobs):,}" for m in placebo_models) + " \\\\\n")

    f.write("\\bottomrule\n\\end{tabular}\n\n")
    f.write("\\medskip\\par\\noindent\\footnotesize\n")
    f.write("\\textit{Notes:} Column~(1) uses per-unit price (cents per ounce) as the "
            "dependent variable. Column~(2) uses total shelf price (dollars). "
            "The negative coefficient in column~(1) and the positive coefficient in "
            "column~(2) confirm that larger packages have lower per-unit prices but "
            "higher total prices, ruling out a mechanical coding artefact. "
            "Standard errors clustered by UPC in parentheses. "
            "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
    f.write("\\end{table}\n")

print("\nWrote tex/placebo_table.tex")
print("\nDone.")
