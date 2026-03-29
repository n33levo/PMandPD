# Product Modification, Robinson-Patman, and Third-Degree Price Discrimination

Sophia Ng and Sohail (Neel) Sarkar, University of Toronto, March 2026.

## What this is

An undergraduate economics paper that models how manufacturers use minor product modifications (package size, formulation, branding) to practice third-degree price discrimination under Robinson-Patman Act constraints. The empirical section provides descriptive evidence consistent with the mechanism using laundry detergent scanner data from Dominick's Finer Foods.

## Data

The raw data is **not** included in this repo (too large). To reproduce:

1. Go to the [Kilts Center for Marketing](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) at Chicago Booth.
2. Download the laundry detergent files:
   - `wlnd.csv` (~456 MB, store-week-UPC movement file)
   - `upclnd.csv` (UPC product lookup, 582 products)
3. Place both CSVs in the repo root.

## Reproducing the analysis

```bash
python code/clean_and_merge.py      # cleans and merges raw data -> scratch/detergent_clean.csv
python code/eda_and_figures.py      # summary stats table + early EDA figures
python code/generate_figures.py     # all 8 publication figures -> figures/
python code/regressions.py          # four OLS models + tex/reg_table.tex
python code/robustness_table.py     # CGM two-way clustering robustness -> tex/robustness_table.tex
(cd tex && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex && cp paper.pdf ../paper.pdf)
```

Requires Python 3.10+ with pandas, numpy, matplotlib, seaborn, statsmodels. A TeX Live or similar LaTeX distribution is needed for compilation.

## Repo structure

```
tex/paper.tex          main manuscript
paper.pdf              compiled PDF
figures/               all 8 publication figures (PNG)
tex/reg_table.tex      regression table (auto-generated)
tex/robustness_table.tex robustness table (auto-generated)
tex/summary_stats.tex  summary statistics table (auto-generated)
code/clean_and_merge.py data pipeline
code/eda_and_figures.py early EDA (exploratory)
code/generate_figures.py final publication figures
code/regressions.py    OLS regressions
code/robustness_table.py two-way clustering robustness checks
regression_results.txt plain-text regression output
```
