# regressions.py
# per-unit price regressions on log(size), store FE, brand FE, sale flag
# uses dominick's laundry detergent data

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(ROOT, "scratch")

df = pd.read_csv(os.path.join(SCRATCH, "detergent_clean.csv"))
print(f"Loaded {len(df):,} rows")

n_stores = df["STORE"].nunique()
n_brands = df["brand"].nunique()
n_upcs = df["UPC"].nunique()
print(f"Stores: {n_stores}, Brands: {n_brands}, UPCs: {n_upcs}")

df["STORE"] = df["STORE"].astype("category")
df["brand"] = df["brand"].astype("category")

# model 1: baseline
print("\nModel 1: Baseline")
m1 = smf.ols("ppu_cents ~ log_size", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["UPC"]})
print(f"  log_size = {m1.params['log_size']:.4f}  "
      f"(SE={m1.bse['log_size']:.4f}, p={m1.pvalues['log_size']:.4f})")
print(f"  R2 = {m1.rsquared:.4f}")

# model 2: + store FE
print("\nModel 2: + Store FE")
m2 = smf.ols("ppu_cents ~ log_size + C(STORE)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["UPC"]})
print(f"  log_size = {m2.params['log_size']:.4f}  "
      f"(SE={m2.bse['log_size']:.4f}, p={m2.pvalues['log_size']:.4f})")
print(f"  R2 = {m2.rsquared:.4f}")

# model 3: + store FE + brand FE
print("\nModel 3: + Store FE + Brand FE")
m3 = smf.ols("ppu_cents ~ log_size + C(STORE) + C(brand)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["UPC"]})
print(f"  log_size = {m3.params['log_size']:.4f}  "
      f"(SE={m3.bse['log_size']:.4f}, p={m3.pvalues['log_size']:.4f})")
print(f"  R2 = {m3.rsquared:.4f}")

# model 4: + sale flag
print("\nModel 4: + Store FE + Brand FE + Sale")
m4 = smf.ols("ppu_cents ~ log_size + C(STORE) + C(brand) + sale_flag",
             data=df).fit(cov_type="cluster", cov_kwds={"groups": df["UPC"]})
print(f"  log_size  = {m4.params['log_size']:.4f}  "
      f"(SE={m4.bse['log_size']:.4f}, p={m4.pvalues['log_size']:.4f})")
print(f"  sale_flag = {m4.params['sale_flag']:.4f}  "
      f"(SE={m4.bse['sale_flag']:.4f}, p={m4.pvalues['sale_flag']:.4f})")
print(f"  R2 = {m4.rsquared:.4f}")

# joint F-test on store FEs (model 2)
store_vars = [v for v in m2.params.index if "STORE" in v]
if store_vars:
    r_matrix = np.zeros((len(store_vars), len(m2.params)))
    for i, sv in enumerate(store_vars):
        j = list(m2.params.index).index(sv)
        r_matrix[i, j] = 1
    wald = m2.wald_test(r_matrix, use_f=True)
    print(f"\nJoint F-test on store FEs: F = {wald.fvalue[0][0]:.2f}, "
          f"p = {wald.pvalue:.6f}")

# export latex regression table
models = [m1, m2, m3, m4]
col_labels = ["(1)", "(2)", "(3)", "(4)"]


def star(p):
    if p < 0.01: return "^{***}"
    if p < 0.05: return "^{**}"
    if p < 0.10: return "^{*}"
    return ""


key_vars = [
    ("log_size",  "$\\ln(\\text{Size})$"),
    ("sale_flag", "Sale indicator"),
]

with open(os.path.join(ROOT, "reg_table.tex"), "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Per-Unit Price Regressions: Dominick's Laundry Detergent}\n")
    f.write("\\label{tab:regression}\n\\small\n")
    f.write("\\begin{tabular}{l" + "c" * len(models) + "}\n\\toprule\n")
    f.write(" & " + " & ".join(col_labels) + " \\\\\n")
    f.write("\\midrule\n")

    for var, nice in key_vars:
        coefs, ses = [], []
        for m in models:
            if var in m.params.index:
                c, s, p = m.params[var], m.bse[var], m.pvalues[var]
                coefs.append(f"${c:.3f}{star(p)}$")
                ses.append(f"$({s:.3f})$")
            else:
                coefs.append("")
                ses.append("")
        f.write(nice + " & " + " & ".join(coefs) + " \\\\\n")
        f.write(" & " + " & ".join(ses) + " \\\\\n[4pt]\n")

    # intercept
    ints, int_ses = [], []
    for m in models:
        c, s, p = m.params["Intercept"], m.bse["Intercept"], m.pvalues["Intercept"]
        ints.append(f"${c:.3f}{star(p)}$")
        int_ses.append(f"$({s:.3f})$")
    f.write("Constant & " + " & ".join(ints) + " \\\\\n")
    f.write(" & " + " & ".join(int_ses) + " \\\\\n")
    f.write("\\midrule\n")

    f.write("Store FE & No & Yes & Yes & Yes \\\\\n")
    f.write("Brand FE & No & No & Yes & Yes \\\\\n")
    f.write("\\midrule\n")

    f.write("$R^2$ & " + " & ".join(f"{m.rsquared:.3f}" for m in models) + " \\\\\n")
    f.write("Adj.\\ $R^2$ & " + " & ".join(f"{m.rsquared_adj:.3f}" for m in models) + " \\\\\n")
    f.write("$N$ & " + " & ".join(f"{int(m.nobs):,}" for m in models) + " \\\\\n")

    f.write("\\bottomrule\n\\end{tabular}\n\n")
    f.write("\\medskip\\par\\noindent\\footnotesize\n")
    f.write("\\textit{Notes:} Dependent variable is per-unit price in cents per ounce. "
            "Standard errors clustered by UPC in parentheses. "
            "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$. "
            f"Store FE: {n_stores} store dummies. "
            f"Brand FE: {n_brands} brand dummies.\n")
    f.write("\\end{table}\n")

print("\nWrote reg_table.tex")

# plain-text summary
with open(os.path.join(ROOT, "regression_results.txt"), "w") as f:
    for label, m in zip(col_labels, models):
        f.write(f"\nModel {label}\n{'='*70}\n")
        f.write(m.summary().as_text() + "\n")
print("Wrote regression_results.txt")
